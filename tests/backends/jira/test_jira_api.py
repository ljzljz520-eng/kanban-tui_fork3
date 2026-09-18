"""Tests for the explicit paginated Jira transport."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from kanban_tui.backends.jira import jira_api
from kanban_tui.backends.jira.jira_api import (
    JiraCancelled,
    JiraPaginationError,
    JiraRateLimitError,
    JiraRequestBudgetExceeded,
    JiraTimeoutError,
    JiraTransportError,
    SearchOptions,
    _parse_retry_after,
    parse_jira_datetime,
    search_issues_paged,
)


def make_issue(key: str, issue_id: int, updated: str | None = None) -> dict:
    return {
        "id": str(issue_id),
        "key": key,
        "fields": {
            "summary": key,
            "updated": updated or "2026-01-01T00:00:00.000+0000",
            "status": {"name": "To Do", "statusCategory": {"name": "To Do"}},
        },
    }


def server_auth(pages: list[dict]) -> MagicMock:
    jql_mock = MagicMock(side_effect=pages)
    return SimpleNamespace(cloud=False, jql=jql_mock)


def fast_options(**kwargs) -> SearchOptions:
    defaults = {"backoff_base": 0.0, "jitter": 0.0, "max_elapsed": 60.0}
    defaults.update(kwargs)
    return SearchOptions(**defaults)


def test_server_pagination_walks_start_at_and_verifies_total():
    pages = [
        {"startAt": 0, "maxResults": 2, "total": 5, "issues": [
            make_issue("PB-1", 1, "2026-01-01T00:00:00.000+0000"),
            make_issue("PB-2", 2, "2026-01-02T00:00:00.000+0000"),
        ]},
        {"startAt": 2, "maxResults": 2, "total": 5, "issues": [
            make_issue("PB-3", 3),
            make_issue("PB-4", 4, "2026-01-03T00:00:00.000+0000"),
        ]},
        {"startAt": 4, "maxResults": 2, "total": 5, "issues": [
            make_issue("PB-5", 5),
        ]},
    ]
    auth = server_auth(pages)

    result = search_issues_paged(auth, "project = PB", fast_options(page_size=2))

    assert result.is_complete
    assert result.total == 5
    assert result.pages_fetched == 3
    assert [issue["key"] for issue in result.issues] == [
        "PB-1",
        "PB-2",
        "PB-3",
        "PB-4",
        "PB-5",
    ]
    # startAt advances by what the server actually reported
    assert [call.kwargs["start"] for call in auth.jql.call_args_list] == [0, 2, 4]
    # newest remote updated timestamp becomes the watermark
    assert result.watermark == parse_jira_datetime("2026-01-03T00:00:00.000+0000")
    assert result.is_cloud is False


def test_server_pagination_deduplicates_overlapping_pages():
    pages = [
        {"startAt": 0, "maxResults": 2, "total": 3, "issues": [
            make_issue("PB-1", 1),
            make_issue("PB-2", 2),
        ]},
        {"startAt": 2, "maxResults": 2, "total": 3, "issues": [
            make_issue("PB-2", 2),
            make_issue("PB-3", 3),
        ]},
    ]
    auth = server_auth(pages)

    result = search_issues_paged(auth, "project = PB", fast_options(page_size=2))

    assert [issue["key"] for issue in result.issues] == ["PB-1", "PB-2", "PB-3"]


def test_server_pagination_raises_when_collected_count_misses_total():
    # Final page lies about its startAt, so the loop ends while total
    # promises more issues; the completeness check must catch it.
    pages = [
        {"startAt": 0, "maxResults": 2, "total": 5, "issues": [
            make_issue("PB-1", 1),
            make_issue("PB-2", 2),
        ]},
        {"startAt": 100, "maxResults": 2, "total": 5, "issues": []},
    ]
    auth = server_auth(pages)

    with pytest.raises(JiraPaginationError, match="collected 2 of 5"):
        search_issues_paged(auth, "project = PB", fast_options(page_size=2))


def test_server_pagination_raises_on_empty_page_before_total():
    pages = [
        {"startAt": 0, "maxResults": 2, "total": 5, "issues": [
            make_issue("PB-1", 1),
        ]},
        {"startAt": 1, "maxResults": 2, "total": 5, "issues": []},
    ]
    auth = server_auth(pages)

    with pytest.raises(JiraPaginationError, match="empty page"):
        search_issues_paged(auth, "project = PB", fast_options(page_size=2))


def test_cloud_pagination_follows_next_page_token():
    pages = [
        {"issues": [make_issue("CL-1", 1)], "nextPageToken": "token-a"},
        {"issues": [make_issue("CL-2", 2)], "nextPageToken": None},
    ]
    enhanced = MagicMock(side_effect=pages)
    auth = SimpleNamespace(cloud=True, enhanced_jql=enhanced)

    result = search_issues_paged(auth, "project = CL", fast_options())

    assert result.is_cloud
    assert result.total is None
    assert result.is_complete
    assert result.pages_fetched == 2
    assert [issue["key"] for issue in result.issues] == ["CL-1", "CL-2"]
    assert enhanced.call_args_list[0].kwargs["nextPageToken"] is None
    assert enhanced.call_args_list[1].kwargs["nextPageToken"] == "token-a"


def http_error(status_code: int, retry_after: str | None = None) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    return requests.exceptions.HTTPError(response=response)


def test_retries_on_429_with_retry_after_and_eventually_succeeds():
    pages = [
        http_error(429, retry_after="0"),
        http_error(429, retry_after="0"),
        {"startAt": 0, "maxResults": 100, "total": 1, "issues": [
            make_issue("PB-1", 1)
        ]},
    ]
    auth = server_auth(pages)

    result = search_issues_paged(
        auth, "project = PB", fast_options(max_retries=3)
    )

    assert result.total == 1
    assert auth.jql.call_count == 3


def test_rate_limit_error_after_exhausted_retries_carries_retry_after():
    auth = server_auth([http_error(429, retry_after="12")])

    with pytest.raises(JiraRateLimitError) as exc_info:
        search_issues_paged(auth, "project = PB", fast_options(max_retries=0))
    assert exc_info.value.retry_after == 12.0


def test_non_retryable_http_status_fails_immediately():
    auth = server_auth([http_error(400)])

    with pytest.raises(JiraTransportError, match="HTTP 400"):
        search_issues_paged(auth, "project = PB", fast_options(max_retries=5))
    assert auth.jql.call_count == 1


def test_timeout_retries_then_raises():
    auth = SimpleNamespace(
        cloud=False,
        jql=MagicMock(side_effect=requests.exceptions.ReadTimeout("slow")),
    )

    with pytest.raises(JiraTimeoutError):
        search_issues_paged(auth, "project = PB", fast_options(max_retries=2))
    assert auth.jql.call_count == 3  # initial + 2 retries


def test_max_pages_budget_is_enforced():
    def page(jql, **kwargs):
        start = kwargs["start"]
        return {
            "startAt": start,
            "maxResults": 2,
            "total": 100,
            "issues": [
                make_issue(f"PB-{start + 1}", start + 1),
                make_issue(f"PB-{start + 2}", start + 2),
            ],
        }

    auth = SimpleNamespace(cloud=False, jql=MagicMock(side_effect=page))

    with pytest.raises(JiraRequestBudgetExceeded) as exc_info:
        search_issues_paged(auth, "project = PB", fast_options(page_size=2, max_pages=2))
    assert exc_info.value.pages_fetched == 2


def test_cancellation_before_first_request(monkeypatch):
    auth = server_auth([])
    # Make any accidental backoff sleep explode loudly
    monkeypatch.setattr(jira_api.time, "sleep", lambda *_: pytest.fail("slept"))

    with pytest.raises(JiraCancelled):
        search_issues_paged(
            auth, "project = PB", fast_options(), is_cancelled=lambda: True
        )
    auth.jql.assert_not_called()


def test_cancellation_during_backoff_aborts():
    # First request gets rate limited, cancel flips while backing off
    calls = {"cancelled": False}

    def side_effect(*args, **kwargs):
        calls["cancelled"] = True
        raise http_error(429, retry_after="30")

    auth = SimpleNamespace(cloud=False, jql=MagicMock(side_effect=side_effect))

    with pytest.raises(JiraCancelled):
        search_issues_paged(
            auth,
            "project = PB",
            fast_options(),
            is_cancelled=lambda: calls["cancelled"],
        )


def test_parse_retry_after_delta_and_http_date():
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("not-a-date") is None
    # Future HTTP date yields a positive delay
    parsed = _parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT")
    assert parsed is not None
    assert parsed >= 0
