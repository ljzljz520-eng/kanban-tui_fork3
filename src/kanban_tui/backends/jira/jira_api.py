"""Explicit, cancellable pagination transport for Jira searches.

The thin :class:`atlassian.jira.Jira` wrapper used here replaces the old
single blocking ``auth.jql(jql)`` call. Jira paginates search results
(``startAt``/``maxResults``/``total`` on Server/Data Center,
``nextPageToken`` on Cloud), so a single request silently truncates any
board larger than the server page size.

``search_issues_paged`` walks every page, verifies that the collected
issue count matches ``total`` and centralises the transport policy:

* per request timeout (configured on the client),
* ``Retry-After`` honoring and exponential backoff on transient failures,
* request budgets (maximum page count / overall elapsed time),
* cooperative cancellation via an ``is_cancelled`` callback.

The async variant runs the blocking pager in a thread executor and turns
asyncio cancellation into the pager's cancel callback, so Textual workers
can abort stale refreshes without freezing the event loop.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import partial

import requests
from atlassian.jira import Jira

CancelCheck = Callable[[], bool] | None

DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_ELAPSED = 300.0

#: HTTP status codes worth retrying (rate limiting / transient failures).
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class JiraTransportError(Exception):
    """Base class for all Jira transport failures."""


class JiraTimeoutError(JiraTransportError):
    """Raised when Jira does not answer in time after all retries."""


class JiraRateLimitError(JiraTransportError):
    """Raised when Jira keeps answering with HTTP 429 after retries."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class JiraRequestBudgetExceeded(JiraTransportError):
    """Raised when the page/time request budget is exhausted."""

    def __init__(self, message: str, *, pages_fetched: int, elapsed: float) -> None:
        super().__init__(message)
        self.pages_fetched = pages_fetched
        self.elapsed = elapsed


class JiraPaginationError(JiraTransportError):
    """Raised when Jira pages are incomplete or contradict ``total``."""


class JiraCancelled(JiraTransportError):
    """Raised when a cancellable search is aborted via its cancel check."""


@dataclass(frozen=True)
class SearchOptions:
    """Policy for paginated Jira searches.

    Attributes:
        page_size: ``maxResults`` requested per page.
        max_pages: Hard cap on fetched pages (request budget). ``None`` is
            bounded by ``max_elapsed`` only.
        request_timeout: Per request timeout in seconds (set on the client).
        max_elapsed: Overall wall clock budget in seconds.
        max_retries: Retries per page on transient failures.
        backoff_base/backoff_factor/backoff_max: Exponential backoff knobs.
        jitter: Relative random jitter added to computed backoff delays.
        fields/expand/validate_query: Forwarded to the Jira search API.
    """

    page_size: int = DEFAULT_PAGE_SIZE
    max_pages: int | None = None
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_elapsed: float = DEFAULT_MAX_ELAPSED
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = 0.5
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    jitter: float = 0.25
    fields: str = "*all"
    expand: str | None = None
    validate_query: str | None = None


@dataclass(frozen=True)
class PagedSearchResult:
    """Verified result of a complete paginated search."""

    issues: tuple[dict, ...]
    total: int | None
    pages_fetched: int
    page_size: int
    watermark: datetime | None
    is_cloud: bool

    @property
    def is_complete(self) -> bool:
        return self.total is None or len(self.issues) == self.total


def parse_jira_datetime(value: str | None) -> datetime | None:
    """Parse a Jira ISO 8601 timestamp (tolerates ``Z``/``+0200``)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None


def authenticate_to_jira(
    base_url: str,
    api_token: str,
    cert_path: str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Jira:
    # Disable the client's own (uncancellable, uncapped) Retry-After sleep
    # so the explicit transport policy below owns retries and budgets.
    return Jira(
        url=base_url,
        token=api_token,
        verify_ssl=cert_path,
        timeout=timeout,
        retry_with_header=False,
        backoff_and_retry=False,
    )


def _raise_if_cancelled(is_cancelled: CancelCheck) -> None:
    if is_cancelled is not None and is_cancelled():
        raise JiraCancelled("Jira request was cancelled")


def _interruptible_sleep(
    seconds: float, is_cancelled: CancelCheck, *, tick: float = 0.1
) -> None:
    """Sleep that stays responsive to cancellation (used during backoff)."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _raise_if_cancelled(is_cancelled)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(tick, remaining))


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta seconds or HTTP date)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at is None or retry_at.tzinfo is None:
        return None
    return max(0.0, (retry_at - datetime.now(retry_at.tzinfo)).total_seconds())


def _backoff_delay(options: SearchOptions, attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, options.backoff_max)
    delay = options.backoff_base * options.backoff_factor**attempt
    if options.jitter:
        delay *= 1.0 + options.jitter * (random.random() * 2.0 - 1.0)
    return min(max(delay, 0.0), options.backoff_max)


def _check_elapsed_budget(
    options: SearchOptions, started_at: float, pages_fetched: int
) -> None:
    elapsed = time.monotonic() - started_at
    if elapsed > options.max_elapsed:
        raise JiraRequestBudgetExceeded(
            f"Jira request budget exceeded: {elapsed:.1f}s > "
            f"{options.max_elapsed:.1f}s after {pages_fetched} page(s)",
            pages_fetched=pages_fetched,
            elapsed=elapsed,
        )


def _request_with_retry(
    request_fn: Callable[[], object],
    options: SearchOptions,
    is_cancelled: CancelCheck,
    started_at: float,
) -> object:
    """Run one blocking Jira request with the transport retry policy."""
    attempt = 0
    while True:
        _raise_if_cancelled(is_cancelled)
        _check_elapsed_budget(options, started_at, pages_fetched=0)
        try:
            return request_fn()
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status = getattr(response, "status_code", None)
            retry_after = (
                _parse_retry_after(response.headers.get("Retry-After"))
                if status == 429 and response is not None
                else None
            )
            if status not in RETRYABLE_STATUS_CODES or attempt >= options.max_retries:
                if status == 429:
                    raise JiraRateLimitError(
                        f"Jira rate limited (HTTP 429) after {attempt} retry attempt(s)",
                        retry_after=retry_after,
                    ) from exc
                raise JiraTransportError(
                    f"Jira request failed with HTTP {status}: {exc}"
                ) from exc
            delay = _backoff_delay(options, attempt, retry_after)
        except requests.exceptions.Timeout as exc:
            if attempt >= options.max_retries:
                raise JiraTimeoutError(
                    f"Jira did not respond within {options.request_timeout}s after "
                    f"{attempt + 1} attempt(s): {exc}"
                ) from exc
            delay = _backoff_delay(options, attempt, None)
        except requests.exceptions.ConnectionError as exc:
            if attempt >= options.max_retries:
                raise JiraTransportError(
                    f"Could not connect to Jira after {attempt + 1} attempt(s): {exc}"
                ) from exc
            delay = _backoff_delay(options, attempt, None)

        _check_elapsed_budget(options, started_at, pages_fetched=0)
        _interruptible_sleep(delay, is_cancelled)
        attempt += 1


def _call_search_page(
    auth: Jira,
    jql: str,
    options: SearchOptions,
    *,
    start_at: int,
    cursor: str | None,
) -> Callable[[], object]:
    if getattr(auth, "cloud", False):
        return partial(
            auth.enhanced_jql,
            jql,
            fields=options.fields,
            nextPageToken=cursor,
            limit=options.page_size,
            expand=options.expand,
        )
    return partial(
        auth.jql,
        jql,
        fields=options.fields,
        start=start_at,
        limit=options.page_size,
        expand=options.expand,
        validate_query=options.validate_query,
    )


def _fetch_page(
    auth: Jira,
    jql: str,
    options: SearchOptions,
    *,
    start_at: int,
    cursor: str | None,
    is_cancelled: CancelCheck,
    started_at: float,
) -> dict:
    page = _request_with_retry(
        _call_search_page(
            auth, jql, options, start_at=start_at, cursor=cursor
        ),
        options,
        is_cancelled,
        started_at,
    )
    if not isinstance(page, dict) or "issues" not in page:
        raise JiraTransportError("Jira returned an unexpected search response")
    return page


def _max_watermark(issues: list[dict]) -> datetime | None:
    latest: datetime | None = None
    for issue in issues:
        updated = parse_jira_datetime((issue.get("fields") or {}).get("updated"))
        if updated is not None and (latest is None or updated > latest):
            latest = updated
    return latest


def search_issues_paged(
    auth: Jira,
    jql: str,
    options: SearchOptions | None = None,
    *,
    is_cancelled: CancelCheck = None,
) -> PagedSearchResult:
    """Fetch every page of a JQL search and verify completeness.

    Server/Data Center: walks ``startAt`` until ``startAt >= total`` and
    verifies ``len(issues) == total``.

    Cloud: follows ``nextPageToken`` until the token is absent.

    Raises:
        JiraPaginationError: if pages end before ``total`` is reached or
            the collected count contradicts ``total``.
        JiraRequestBudgetExceeded: on page/elapsed budget exhaustion.
        JiraCancelled: when ``is_cancelled`` returns ``True``.
    """
    options = options or SearchOptions()
    started_at = time.monotonic()
    is_cloud = bool(getattr(auth, "cloud", False))

    collected: list[dict] = []
    seen_keys: set[str] = set()
    total: int | None = None
    start_at = 0
    cursor: str | None = None
    pages_fetched = 0

    while True:
        _raise_if_cancelled(is_cancelled)
        _check_elapsed_budget(options, started_at, pages_fetched)
        if options.max_pages is not None and pages_fetched >= options.max_pages:
            raise JiraRequestBudgetExceeded(
                f"Jira page budget exceeded: more than {options.max_pages} page(s) "
                f"required for this JQL query",
                pages_fetched=pages_fetched,
                elapsed=time.monotonic() - started_at,
            )

        page = _fetch_page(
            auth,
            jql,
            options,
            start_at=start_at,
            cursor=cursor,
            is_cancelled=is_cancelled,
            started_at=started_at,
        )
        pages_fetched += 1
        page_issues = page.get("issues") or []

        if not is_cloud and total is None:
            total = page.get("total")

        for issue in page_issues:
            key = issue.get("key") or issue.get("id")
            if key is None or key in seen_keys:
                continue
            seen_keys.add(key)
            collected.append(issue)

        if is_cloud:
            cursor = page.get("nextPageToken")
            if not cursor:
                break
            continue

        page_start = page.get("startAt", start_at)
        # Advance by what the server actually reported, not by the asked size:
        # Jira may cap maxResults below the requested value.
        start_at = int(page_start) + len(page_issues)

        if not page_issues:
            if total is None or start_at >= int(total):
                break
            raise JiraPaginationError(
                f"Jira returned an empty page at startAt={page_start} but total="
                f"{total} promises more issues"
            )

        if total is not None and start_at >= int(total):
            break

    if not is_cloud and total is not None and len(collected) != int(total):
        raise JiraPaginationError(
            f"Incomplete Jira result set: collected {len(collected)} of {total} "
            f"issue(s) across {pages_fetched} page(s)"
        )

    return PagedSearchResult(
        issues=tuple(collected),
        total=total,
        pages_fetched=pages_fetched,
        page_size=options.page_size,
        watermark=_max_watermark(collected),
        is_cloud=is_cloud,
    )


async def search_issues_paged_async(
    auth: Jira,
    jql: str,
    options: SearchOptions | None = None,
) -> PagedSearchResult:
    """Async wrapper around :func:`search_issues_paged`.

    The blocking pager runs in a thread executor. When the awaiting worker
    is cancelled the pager's ``is_cancelled`` callback flips, so the next
    page boundary aborts instead of continuing to hit Jira.
    """
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()
    pager = partial(
        search_issues_paged,
        auth,
        jql,
        options,
        is_cancelled=cancel_event.is_set,
    )
    future = loop.run_in_executor(None, pager)
    try:
        return await future
    except asyncio.CancelledError:
        # Stop the background pager before its next request. The in-flight
        # request is left to finish on its own timeout; we do not await it
        # here so cancellation stays responsive.
        cancel_event.set()
        future.cancel()
        raise
    except JiraCancelled as exc:  # pragma: no cover - defensive mapping
        raise asyncio.CancelledError(str(exc)) from exc


def get_issue_state(
    auth: Jira,
    issue_key: str,
    options: SearchOptions | None = None,
    *,
    is_cancelled: CancelCheck = None,
) -> dict:
    """Fetch the lightweight remote state of one issue.

    Returns a dict with ``key``, ``status`` and a parsed ``updated``
    timestamp. Used to carry the remote ``updated`` watermark into status
    transitions so stale snapshots can be detected.
    """
    options = options or SearchOptions()
    started_at = time.monotonic()
    data = _request_with_retry(
        partial(auth.issue, issue_key, fields="updated,status"),
        options,
        is_cancelled,
        started_at,
    )
    if not isinstance(data, dict):
        raise JiraTransportError(f"Jira returned an unexpected issue for {issue_key}")
    fields = data.get("fields") or {}
    status = fields.get("status") or {}
    return {
        "key": issue_key,
        "status": status.get("name"),
        "updated": parse_jira_datetime(fields.get("updated")),
    }


def get_jql(
    auth: Jira,
    jql: str,
    *,
    limit: int | None = None,
    options: SearchOptions | None = None,
) -> dict:
    """Fetch a single JQL page (retry/timeout aware).

    Backward-compatible replacement for the raw ``auth.jql(jql)`` call.
    Callers that need the whole result set must use
    :func:`search_issues_paged` instead.
    """
    options = replace(options or SearchOptions(), max_pages=1)
    if limit is not None:
        options = replace(options, page_size=limit)
    return _fetch_page(
        auth,
        jql,
        options,
        start_at=0,
        cursor=None,
        is_cancelled=None,
        started_at=time.monotonic(),
    )


async def get_jql_async(auth: Jira, jql: str, *, limit: int | None = None):
    """Get one JQL page asynchronously in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(get_jql, auth, jql, limit=limit))


def get_transitions(auth: Jira, issue_key: str):
    """Get available transitions for a specific issue"""
    return auth.get_issue_transitions(issue_key)


async def get_transitions_async(auth: Jira, issue_key: str):
    """Get available transitions asynchronously in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(get_transitions, auth, issue_key))


def set_issue_status(auth: Jira, issue_key: str, transition_id: int) -> dict:
    """Execute a transition to change issue status.

    Uses set_issue_status_by_transition_id which accepts a numeric
    transition ID, as opposed to auth.set_issue_status which expects
    a status *name* string.
    """
    try:
        auth.set_issue_status_by_transition_id(issue_key, transition_id)
        return {"success": True, "message": f"Transitioned {issue_key}"}
    except Exception as e:
        return {"success": False, "message": str(e)}
