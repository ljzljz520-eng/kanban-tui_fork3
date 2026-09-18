from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from atlassian import Jira
from pydantic import ValidationError

from kanban_tui.backends.auth import AuthSettings, init_auth_file
from kanban_tui.backends.base import Backend
from kanban_tui.backends.jira.jira_api import (
    JiraTransportError,
    SearchOptions,
    authenticate_to_jira,
    get_issue_state,
    get_transitions,
    search_issues_paged,
    search_issues_paged_async,
    set_issue_status,
)
from kanban_tui.backends.jira.models import JiraBoardSnapshot, JiraIssue
from kanban_tui.classes.board import Board
from kanban_tui.classes.category import Category
from kanban_tui.classes.column import Column
from kanban_tui.classes.task import Task
from kanban_tui.config import JiraBackendSettings, JqlEntry
from kanban_tui.constants import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class JiraBackend(Backend):
    settings: JiraBackendSettings
    auth: Jira = field(init=False)
    auth_settings: AuthSettings = field(init=False)

    def __post_init__(self):
        init_auth_file(self.settings.auth_file_path)
        self.auth_settings = AuthSettings()
        self.get_authentication()
        # board_id -> last-known-good, fully verified snapshot
        self._snapshots: dict[int, JiraBoardSnapshot] = {}
        self._cache_lock = threading.Lock()
        self._cache_dir = Path(
            self.settings.snapshot_cache_path or (DATA_DIR / "jira_snapshots")
        )

    def get_authentication(self):
        self.auth = authenticate_to_jira(
            self.settings.base_url,
            self.api_key,
            self.cert_path,
            timeout=self.settings.request_timeout,
        )

    # ------------------------------------------------------------------
    # Snapshot transport / publishing
    # ------------------------------------------------------------------

    def _search_options(self) -> SearchOptions:
        return SearchOptions(
            page_size=self.settings.page_size,
            request_timeout=self.settings.request_timeout,
        )

    def _get_jql_entry(self, board_id: int) -> JqlEntry:
        entry = next(
            (entry for entry in self.settings.jqls if entry.id == board_id), None
        )
        if entry is None:
            raise ValueError(f"No JQL board configured for board_id={board_id}")
        return entry

    @staticmethod
    def _snapshot_matches(snapshot: JiraBoardSnapshot, entry: JqlEntry) -> bool:
        return (
            snapshot.jql == entry.jql
            and dict(snapshot.column_mapping) == dict(entry.column_mapping)
        )

    def _build_snapshot(self, entry: JqlEntry, result) -> JiraBoardSnapshot:
        """Convert a verified paged result into an immutable snapshot.

        Dependency resolution runs over the *complete* issue set, so links
        crossing server page boundaries are resolved as well.
        """
        issues = list(result.issues)
        tasks = [
            self._jira_issue_to_task(issue_data, board_id=entry.id)
            for issue_data in issues
        ]
        unresolved_link_count = self._resolve_issue_dependencies(tasks, issues)
        return JiraBoardSnapshot(
            board_id=entry.id,
            jql=entry.jql,
            column_mapping=dict(entry.column_mapping),
            tasks=tuple(tasks),
            total=result.total,
            pages_fetched=result.pages_fetched,
            watermark=result.watermark,
            fetched_at=datetime.now(),
            unresolved_link_count=unresolved_link_count,
        )

    def _publish_snapshot(self, snapshot: JiraBoardSnapshot) -> None:
        """Atomically publish a complete snapshot to memory and disk cache."""
        with self._cache_lock:
            self._snapshots[snapshot.board_id] = snapshot
        self._persist_snapshot(snapshot)

    def refresh_board_snapshot(
        self, board_id: int, *, is_cancelled=None
    ) -> JiraBoardSnapshot:
        """Fetch and verify all pages, then publish a snapshot.

        Raises a :class:`JiraTransportError` subclass on failure. A failed
        refresh never replaces the previously published last-known-good
        snapshot.
        """
        entry = self._get_jql_entry(board_id)
        result = search_issues_paged(
            self.auth,
            entry.jql,
            self._search_options(),
            is_cancelled=is_cancelled,
        )
        snapshot = self._build_snapshot(entry, result)
        self._publish_snapshot(snapshot)
        return snapshot

    async def arefresh_board(self, board_id: int) -> JiraBoardSnapshot:
        """Async refresh path for the Textual worker (cancellable)."""
        entry = self._get_jql_entry(board_id)
        result = await search_issues_paged_async(
            self.auth, entry.jql, self._search_options()
        )
        snapshot = self._build_snapshot(entry, result)
        self._publish_snapshot(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Last-known-good cache (memory + optional disk)
    # ------------------------------------------------------------------

    def _cache_file(self, board_id: int) -> Path:
        return self._cache_dir / f"jira_snapshot_board_{board_id}.json"

    def _persist_snapshot(self, snapshot: JiraBoardSnapshot) -> None:
        try:
            cache_dir = self._cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "board_id": snapshot.board_id,
                "jql": snapshot.jql,
                "column_mapping": dict(snapshot.column_mapping),
                "total": snapshot.total,
                "pages_fetched": snapshot.pages_fetched,
                "watermark": snapshot.watermark.isoformat()
                if snapshot.watermark
                else None,
                "fetched_at": snapshot.fetched_at.isoformat(),
                "unresolved_link_count": snapshot.unresolved_link_count,
                "tasks": [task.model_dump(mode="json") for task in snapshot.tasks],
            }
            target = self._cache_file(snapshot.board_id)
            tmp_file = target.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp_file, target)
        except OSError:
            logger.debug("Could not persist Jira snapshot cache", exc_info=True)

    def _load_disk_snapshot(self, entry: JqlEntry) -> JiraBoardSnapshot | None:
        path = self._cache_file(entry.id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("jql") != entry.jql:
                return None
            if dict(payload.get("column_mapping") or {}) != dict(
                entry.column_mapping
            ):
                return None
            tasks = tuple(
                Task.model_validate(item) for item in payload.get("tasks", [])
            )
            watermark = payload.get("watermark")
            watermark_dt = datetime.fromisoformat(watermark) if watermark else None
            return JiraBoardSnapshot(
                board_id=entry.id,
                jql=entry.jql,
                column_mapping=dict(entry.column_mapping),
                tasks=tasks,
                total=payload.get("total"),
                pages_fetched=int(payload.get("pages_fetched", 0)),
                watermark=watermark_dt,
                fetched_at=datetime.fromisoformat(payload["fetched_at"]),
                unresolved_link_count=int(payload.get("unresolved_link_count", 0)),
            )
        except (OSError, ValueError, KeyError, ValidationError):
            logger.debug("Ignoring unreadable Jira snapshot cache %s", path)
            return None

    def get_cached_snapshot(self, board_id: int) -> JiraBoardSnapshot | None:
        """Return the last-known-good snapshot without touching the network."""
        with self._cache_lock:
            snapshot = self._snapshots.get(board_id)
        if snapshot is not None:
            return snapshot

        entry = next(
            (entry for entry in self.settings.jqls if entry.id == board_id), None
        )
        if entry is None:
            return None
        snapshot = self._load_disk_snapshot(entry)
        if snapshot is not None:
            with self._cache_lock:
                self._snapshots.setdefault(board_id, snapshot)
        return snapshot

    def get_cached_tasks(self, board_id: int) -> list[Task]:
        snapshot = self.get_cached_snapshot(board_id)
        return list(snapshot.tasks) if snapshot is not None else []

    def get_cached_active_tasks(self) -> list[Task]:
        return self.get_cached_tasks(self.settings.active_jql)

    def invalidate_snapshot(self, board_id: int) -> None:
        with self._cache_lock:
            self._snapshots.pop(board_id, None)
        try:
            self._cache_file(board_id).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove snapshot cache", exc_info=True)

    # Queries
    def get_boards(self) -> list[Board]:
        """Return a virtual board representing the active JQL query results"""
        if not self.settings.jqls:
            return []

        # Create a virtual board from the JQL query
        return [
            Board(
                board_id=entry.id,
                name=entry.name,
                icon=":mag:",
                creation_date=datetime.now(),
                reset_column=1,  # To Do
                start_column=2,  # In Progress
                finish_column=3,  # Done
            )
            for entry in self.settings.jqls
        ]

    def get_all_categories(self) -> list[Category]:
        """Jira backend doesn't support categories (could map from labels later)"""
        return []

    def get_board_infos(self) -> list[dict]:
        """Return info about the virtual Jira boards.

        Purely cache/local-data based: opening the board overview must not
        trigger one blocking JQL query per board. Boards without a
        last-known-good snapshot report ``None`` counts instead of fetching.
        """
        boards = self.get_boards()
        if not boards:
            return []

        board_infos = []
        for board in boards:
            snapshot = self.get_cached_snapshot(board.board_id)
            board_tasks = list(snapshot.tasks) if snapshot is not None else []

            board_info_dict = {
                "board_id": board.board_id,
                "amount_tasks": len(board_tasks) if snapshot is not None else None,
                "amount_columns": len(self.get_columns(board_id=board.board_id)),
                "next_due": min(
                    (t.due_date for t in board_tasks if t.due_date), default=None
                ),
                "stale": snapshot is None,
            }
            board_infos.append(board_info_dict)

        return board_infos

    def get_columns(self, board_id: int | None = None) -> list[Column]:
        """Return columns based on status mapping"""
        # Create columns from unique status-to-column mappings
        if not board_id and self.active_board:
            board_id = self.active_board.board_id
        if board_id is None:
            return []

        # Get the column mapping for this specific board
        status_column_map = self._get_column_mapping_for_board(board_id)

        columns = []
        for status, column_id in status_column_map.items():
            columns.append(
                Column(
                    column_id=column_id,
                    name=status,
                    visible=True,
                    position=column_id - 1,
                    board_id=board_id,
                )
            )

        return columns

    def get_column_by_id(self, column_id: int) -> Column | None:
        """Return a single column by its ID"""
        return next(
            (column for column in self.get_columns() if column_id == column.column_id),
            None,
        )

    def get_tasks_by_board_id(self, board_id: int) -> list[Task]:
        """Return the board's tasks, refreshing the snapshot when needed.

        Serves last-known-good on transport failures instead of raising, so
        a rate limited/offline Jira never wipes the rendered board.
        """
        entry = self._get_jql_entry(board_id)
        snapshot = self.get_cached_snapshot(board_id)
        if snapshot is None or not self._snapshot_matches(snapshot, entry):
            try:
                snapshot = self.refresh_board_snapshot(board_id)
            except JiraTransportError:
                if snapshot is None:
                    raise
        return list(snapshot.tasks)

    def get_tasks_on_active_board(self, *, force_refresh: bool = False) -> list[Task]:
        """Return active board tasks.

        Fresh CLI processes hold no cache and transparently fetch; the
        running app uses the non-blocking cached/refresh paths instead.
        """
        board_id = self.settings.active_jql
        if force_refresh:
            return self.get_tasks_by_board_id(board_id=board_id)

        entry = next(
            (entry for entry in self.settings.jqls if entry.id == board_id), None
        )
        snapshot = self.get_cached_snapshot(board_id)
        if (
            snapshot is not None
            and entry is not None
            and self._snapshot_matches(snapshot, entry)
        ):
            return list(snapshot.tasks)
        return self.get_tasks_by_board_id(board_id=board_id)

    def get_task_by_id(self, task_id: int) -> Task | None:
        """Fetch a single Jira issue by ID"""
        tasks = self.get_tasks_by_ids([task_id])
        return tasks[0] if tasks else None

    def get_tasks_by_ids(self, task_ids: list[int]) -> list[Task]:
        """Fetch specific issues, preferring cached snapshots.

        Cache misses are fetched in a *single* batched JQL query instead of
        one request per issue.
        """
        if not task_ids:
            return []

        wanted = list(dict.fromkeys(int(task_id) for task_id in task_ids))
        found: dict[int, Task] = {}

        with self._cache_lock:
            snapshots = list(self._snapshots.values())
        for snapshot in snapshots:
            for task in snapshot.tasks:
                if task.task_id in wanted and task.task_id not in found:
                    found[task.task_id] = task

        remaining = [task_id for task_id in wanted if task_id not in found]
        if remaining:
            id_list = ", ".join(f'"{task_id}"' for task_id in remaining)
            try:
                result = search_issues_paged(
                    self.auth, f"id in ({id_list})", self._search_options()
                )
            except JiraTransportError:
                logger.debug(
                    "Could not fetch issues %s from Jira", remaining, exc_info=True
                )
                result = None
            if result is not None:
                board_id = self.settings.active_jql
                for issue_data in result.issues:
                    task = self._jira_issue_to_task(
                        issue_data, board_id=board_id
                    )
                    found.setdefault(task.task_id, task)

        return [found[task_id] for task_id in wanted if task_id in found]

    # Helper methods

    def _get_active_jql_entry(self):
        """Get the active JQL entry"""
        if not self.settings.jqls:
            return None

        for entry in self.settings.jqls:
            if entry.id == self.settings.active_jql:
                return entry

        # Fallback to first entry
        return self.settings.jqls[0] if self.settings.jqls else None

    def _jira_issue_to_task(
        self, issue_data: dict, board_id: int | None = None
    ) -> Task:
        """Convert Jira issue dict to Task model"""
        jira_issue = JiraIssue(**issue_data)

        # Map Jira status to column using board-specific mapping
        column_id = self._status_to_column(jira_issue.status, board_id)

        # Compute start/finish dates based on status category
        start_date = None
        finish_date = None

        status_category = jira_issue.status_category.lower()
        if status_category == "in progress":
            start_date = jira_issue.created
        elif status_category == "done":
            start_date = jira_issue.created
            finish_date = (
                jira_issue.resolution_date or jira_issue.updated or datetime.now()
            )

        # Use Jira's numeric ID as task_id (must be int)
        # Store the Jira key in metadata
        task_id = int(jira_issue.id)

        # Build metadata with Jira-specific fields
        metadata = {
            "jira_key": jira_issue.key,
            "assignee": jira_issue.assignee,
            "assignee_email": jira_issue.assignee_email,
            "reporter": jira_issue.reporter,
            "priority": jira_issue.priority,
            "issue_type": jira_issue.issue_type,
            "labels": jira_issue.labels,
            "components": jira_issue.components,
            "status": jira_issue.status,
            "status_category": jira_issue.status_category,
            "updated": jira_issue.updated.isoformat() if jira_issue.updated else None,
            "resolution": jira_issue.resolution,
            "backend_source": "jira",
            # Links pointing to issues outside the query scope, populated by
            # _resolve_issue_dependencies; each entry is
            # {"id", "key", "relation" ("blocks"/"depends_on"), "direction"}.
            "unresolved_links": [],
        }

        return Task(
            task_id=task_id,
            title=f"{jira_issue.key}\n{jira_issue.summary}",
            column=column_id,
            creation_date=jira_issue.created,
            start_date=start_date,
            finish_date=finish_date,
            due_date=jira_issue.due_date,
            description=jira_issue.description,
            category=None,  # Could map from labels/components later
            blocked_by=[],  # Will be populated by _resolve_issue_dependencies
            blocking=[],
            metadata=metadata,
        )

    def _get_column_mapping_for_board(self, board_id: int | None) -> dict[str, int]:
        """Get the column mapping for a specific board"""
        if board_id is None:
            return {}

        # Find the JQL entry for this board
        jql_entry = next(
            (entry for entry in self.settings.jqls if entry.id == board_id), None
        )

        if jql_entry and jql_entry.column_mapping:
            return jql_entry.column_mapping

        return {}

    def _status_to_column(self, status: str, board_id: int | None = None) -> int:
        """Map Jira status to kanban column"""
        column_mapping = self._get_column_mapping_for_board(board_id)
        return column_mapping.get(status, 1)  # Default to first column

    @staticmethod
    def _comparable_datetime(value: datetime | None) -> datetime | None:
        """Normalize aware/naive datetimes to naive UTC for comparison."""
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    def _resolve_issue_dependencies(
        self, tasks: list[Task], issues: list[dict]
    ) -> int:
        """Resolve Jira issue links over the complete issue set.

        Links whose target issue is not part of the fetched snapshot cannot
        be represented by an in-board task id; they are recorded as
        ``metadata["unresolved_links"]`` instead of being silently dropped.

        Returns:
            Number of unresolved links encountered.
        """
        key_to_task: dict[str, Task] = {}
        id_to_task: dict[str, Task] = {}

        for issue_data, task in zip(issues, tasks, strict=False):
            if issue_data.get("key") is not None:
                key_to_task[issue_data["key"]] = task
            if issue_data.get("id") is not None:
                id_to_task[str(issue_data["id"])] = task

        unresolved_count = 0

        def find_target(ref) -> Task | None:
            if not isinstance(ref, dict):
                return None
            ref_id = ref.get("id")
            if ref_id is not None:
                target = id_to_task.get(str(ref_id))
                if target is not None:
                    return target
            ref_key = ref.get("key")
            if isinstance(ref_key, str):
                return key_to_task.get(ref_key)
            return None

        def append_unique(values: list[int], value: int) -> None:
            if value not in values:
                values.append(value)

        for issue_data, task in zip(issues, tasks, strict=False):
            issue_links = (issue_data.get("fields") or {}).get("issuelinks", [])

            for link in issue_links:
                link_type = link.get("type", {})
                link_type_name = link_type.get("name", "").lower()
                is_block = "block" in link_type_name
                is_depend = "depend" in link_type_name
                if not (is_block or is_depend):
                    continue
                relation = "blocks" if is_block else "depends_on"

                if "outwardIssue" in link:
                    outward_task = find_target(link["outwardIssue"])
                    if outward_task is None:
                        task.metadata["unresolved_links"].append(
                            {
                                "id": (link["outwardIssue"] or {}).get("id"),
                                "key": (link["outwardIssue"] or {}).get("key"),
                                "relation": relation,
                                "direction": "outward",
                            }
                        )
                        unresolved_count += 1
                        continue
                    if is_block:
                        # Current issue blocks the outward issue
                        append_unique(task.blocking, outward_task.task_id)
                    else:
                        # Current issue depends on the outward issue
                        append_unique(task.blocked_by, outward_task.task_id)

                if "inwardIssue" in link:
                    inward_task = find_target(link["inwardIssue"])
                    if inward_task is None:
                        task.metadata["unresolved_links"].append(
                            {
                                "id": (link["inwardIssue"] or {}).get("id"),
                                "key": (link["inwardIssue"] or {}).get("key"),
                                "relation": relation,
                                "direction": "inward",
                            }
                        )
                        unresolved_count += 1
                        continue
                    if is_block:
                        # Inward issue blocks current issue
                        append_unique(task.blocked_by, inward_task.task_id)
                    else:
                        # Inward issue depends on current issue
                        append_unique(task.blocking, inward_task.task_id)

        return unresolved_count

    @property
    def active_board(self) -> Board | None:
        boards = self.get_boards()
        if self.settings.active_jql:
            for board in boards:
                if board.board_id == self.settings.active_jql:
                    return board
        # Default to first board
        return boards[0]

    @property
    def api_key(self) -> str:
        return self.auth_settings.jira.api_key

    @property
    def cert_path(self) -> str:
        return self.auth_settings.jira.cert_path

    # Read-only backend - these methods raise NotImplementedError

    def create_new_task(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Create tasks in Jira directly."
        )

    def update_task_entry(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Update tasks in Jira directly."
        )

    def delete_task(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Delete tasks in Jira directly."
        )

    def update_task_status(
        self,
        new_task: Task,
        target_position: int | None = None,
        append_mode=None,
    ) -> dict[str, bool | str]:
        """Update Jira issue status by finding a transition whose target
        status maps to the same column the task was moved to.

        Carries the snapshot's remote ``updated`` watermark: if the issue
        changed on the server after the snapshot was fetched, the move is
        rejected with ``conflict=True`` so the UI can force a refresh
        instead of transitioning stale state.

        Args:
            new_task: Task with updated column information

        Returns:
            dict with 'success' (bool), optionally 'conflict' (bool) and
            'message' (str) keys
        """
        # target_position / append_mode are sqlite-specific and intentionally ignored.
        _ = target_position, append_mode
        jira_key = new_task.metadata.get("jira_key")
        if not jira_key:
            return {
                "success": False,
                "message": "Task does not have a Jira key in metadata",
            }

        board_id = self.active_board.board_id if self.active_board else None
        target_column = new_task.column
        snapshot = (
            self.get_cached_snapshot(board_id) if board_id is not None else None
        )

        try:
            remote_state = get_issue_state(
                self.auth, jira_key, self._search_options()
            )
        except JiraTransportError as e:
            return {
                "success": False,
                "message": f"Failed to verify remote issue state: {e!s}",
            }

        remote_updated = self._comparable_datetime(remote_state.get("updated"))
        snapshot_watermark = self._comparable_datetime(
            snapshot.watermark if snapshot is not None else None
        )
        if (
            remote_updated is not None
            and snapshot_watermark is not None
            and remote_updated > snapshot_watermark
        ):
            return {
                "success": False,
                "conflict": True,
                "message": (
                    f"{jira_key} changed on the server after the last refresh. "
                    "Press r to reload before moving it."
                ),
            }

        try:
            transitions = get_transitions(self.auth, jira_key)
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to fetch transitions: {e!s}",
            }

        # For each transition, map its target status back to a column ID
        # using the column mapping lookup, then compare column IDs
        # (int == int) instead of comparing status name strings.
        column_mapping = self._get_column_mapping_for_board(board_id)
        transition_id = None
        available = []

        for transition in transitions:
            if not isinstance(transition, dict):
                continue

            to_status = transition.get("to", "")
            column_for_transition = column_mapping.get(to_status)
            available.append(f"{to_status} (col {column_for_transition})")

            if (
                column_for_transition is not None
                and column_for_transition == target_column
            ):
                transition_id = transition.get("id")
                break

        if transition_id is None:
            return {
                "success": False,
                "message": f"No transition available to column {target_column}. Available: {', '.join(available)}",
            }

        result = set_issue_status(self.auth, jira_key, transition_id)
        if result.get("success"):
            result["watermark"] = (
                remote_state["updated"].isoformat()
                if remote_state.get("updated")
                else None
            )
        return result

    def create_new_board(
        self, name: str, jql: str, column_mapping: dict[str, int] | None = None
    ) -> int:
        new_id = self.settings.jqls[-1].id + 1 if self.settings.jqls else 1

        new_jql = JqlEntry(
            id=new_id, name=name, jql=jql, column_mapping=column_mapping or {}
        )
        self.settings.jqls.append(new_jql)
        return new_id

    def update_board_entry(self, *args, **kwargs):
        raise NotImplementedError("Jira backend is read-only. Update boards in config.")

    def delete_board(self, board_id: int):
        jql_to_delete = None
        for jql in self.settings.jqls:
            if board_id == jql.id:
                jql_to_delete = jql

        if jql_to_delete is not None:
            self.settings.jqls.remove(jql_to_delete)
            self.invalidate_snapshot(board_id)

    def create_new_column(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Columns are mapped from Jira statuses."
        )

    def update_column_visibility(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Update column visibility in config."
        )

    def switch_column_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Column positions are fixed."
        )

    def update_column_name(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Column names come from Jira statuses."
        )

    def delete_column(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Columns cannot be deleted."
        )

    def create_new_category(self, *args, **kwargs):
        raise NotImplementedError("Jira backend doesn't support categories.")

    def update_category_entry(self, *args, **kwargs):
        raise NotImplementedError("Jira backend doesn't support categories.")

    def delete_category(self, *args, **kwargs):
        raise NotImplementedError("Jira backend doesn't support categories.")

    def create_task_dependency(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Create dependencies in Jira directly."
        )

    def delete_task_dependency(self, *args, **kwargs):
        raise NotImplementedError(
            "Jira backend is read-only. Delete dependencies in Jira directly."
        )
