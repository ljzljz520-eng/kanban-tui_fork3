from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from kanban_tui.backends.jira import backend as backend_module
from kanban_tui.backends.jira.backend import JiraBackend
from kanban_tui.backends.jira.jira_api import (
    JiraTransportError,
    PagedSearchResult,
)
from kanban_tui.config import JiraBackendSettings, JqlEntry, Settings

WATERMARK = datetime(2026, 1, 1, tzinfo=UTC)
COLUMN_MAPPING = {"To Do": 1, "In Progress": 2, "Done": 3}


def make_issue(
    issue_id: int,
    key: str,
    *,
    status: str = "To Do",
    updated: str = "2026-01-01T00:00:00.000+0000",
    links: list[dict] | None = None,
) -> dict:
    return {
        "id": str(issue_id),
        "key": key,
        "fields": {
            "summary": key,
            "created": "2026-01-01T00:00:00.000+0000",
            "updated": updated,
            "status": {"name": status, "statusCategory": {"name": status}},
            "issuelinks": links or [],
        },
    }


def blocks_outward(target_id: int, target_key: str) -> dict:
    return {
        "type": {"name": "Blocks"},
        "outwardIssue": {"id": str(target_id), "key": target_key},
    }


def blocks_inward(target_id: int, target_key: str) -> dict:
    return {
        "type": {"name": "Blocks"},
        "inwardIssue": {"id": str(target_id), "key": target_key},
    }


@pytest.fixture
def jira_settings(tmp_path, test_auth_path) -> JiraBackendSettings:
    return JiraBackendSettings(
        base_url="http://localhost:8080",
        auth_file_path=test_auth_path,
        snapshot_cache_path=(tmp_path / "snapshots").as_posix(),
        jqls=[
            JqlEntry(
                id=1,
                name="Board One",
                jql="project = PB",
                column_mapping=dict(COLUMN_MAPPING),
            )
        ],
        active_jql=1,
    )


@pytest.fixture
def jira_backend(
    jira_settings: JiraBackendSettings, monkeypatch: pytest.MonkeyPatch
) -> JiraBackend:
    monkeypatch.setenv("KANBAN_TUI_AUTH_FILE", jira_settings.auth_file_path)
    backend = JiraBackend(settings=jira_settings)
    backend.auth = MagicMock()
    return backend


def patch_pager(
    monkeypatch: pytest.MonkeyPatch,
    issues: list[dict],
    *,
    watermark: datetime | None = WATERMARK,
    pages_fetched: int = 1,
) -> MagicMock:
    result = PagedSearchResult(
        issues=tuple(issues),
        total=len(issues),
        pages_fetched=pages_fetched,
        page_size=100,
        watermark=watermark,
        is_cloud=False,
    )
    pager = MagicMock(return_value=result)
    monkeypatch.setattr(backend_module, "search_issues_paged", pager)
    return pager


def test_init_backend(test_jira_config: Settings, test_auth_path):
    import os

    os.environ["KANBAN_TUI_AUTH_FILE"] = test_auth_path
    backend = JiraBackend(settings=test_jira_config.backend.jira_settings)
    assert backend.settings.base_url == "http://localhost:8080"
    assert backend.settings.auth_file_path == test_auth_path
    from pathlib import Path

    assert Path(test_auth_path).exists()


def test_refresh_publishes_snapshot_only_after_full_fetch(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    issues = [make_issue(10, "PB-1"), make_issue(20, "PB-2")]
    patch_pager(monkeypatch, issues, pages_fetched=2)

    snapshot = jira_backend.refresh_board_snapshot(1)

    assert snapshot.is_complete
    assert snapshot.pages_fetched == 2
    assert len(snapshot.tasks) == 2
    assert jira_backend.get_cached_snapshot(1) is snapshot
    assert len(jira_backend.get_cached_active_tasks()) == 2
    assert jira_backend._cache_file(1).exists()


def test_failed_refresh_keeps_last_known_good(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_pager(monkeypatch, [make_issue(10, "PB-1")])
    good = jira_backend.refresh_board_snapshot(1)

    failing_pager = MagicMock(side_effect=JiraTransportError("rate limited"))
    monkeypatch.setattr(backend_module, "search_issues_paged", failing_pager)

    with pytest.raises(JiraTransportError):
        jira_backend.refresh_board_snapshot(1)

    # memory last-known-good untouched
    assert jira_backend.get_cached_snapshot(1) is good
    # read path serves the stale snapshot instead of raising
    tasks = jira_backend.get_tasks_by_board_id(board_id=1)
    assert [t.task_id for t in tasks] == [10]


def test_failed_refresh_without_cache_raises(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        backend_module,
        "search_issues_paged",
        MagicMock(side_effect=JiraTransportError("offline")),
    )
    with pytest.raises(JiraTransportError):
        jira_backend.get_tasks_by_board_id(board_id=1)


def test_disk_cache_roundtrip_and_stale_jql_invalidation(
    jira_backend: JiraBackend,
    jira_settings: JiraBackendSettings,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_pager(monkeypatch, [make_issue(10, "PB-1")])
    jira_backend.refresh_board_snapshot(1)
    cache_file = jira_backend._cache_file(1)
    assert cache_file.exists()

    # A fresh process-like instance loads last-known-good from disk
    second = JiraBackend(settings=jira_settings)
    second.auth = MagicMock()
    second._snapshots.clear()
    loaded = second.get_cached_snapshot(1)
    assert loaded is not None
    assert [t.task_id for t in loaded.tasks] == [10]

    # Changed JQL invalidates the on-disk snapshot
    jira_settings.jqls[0].jql = "project = OTHER"
    second._snapshots.clear()
    assert second.get_cached_snapshot(1) is None


def test_dependencies_resolved_across_pages_with_unresolved_links(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    # A blocks B (link mirrored on both sides as Jira returns it), and
    # C blocks PB-99 which lives outside this JQL result set.
    issues = [
        make_issue(10, "PB-1", links=[blocks_outward(20, "PB-2")]),
        make_issue(
            20,
            "PB-2",
            status="In Progress",
            links=[blocks_inward(10, "PB-1")],
        ),
        make_issue(30, "PB-3", links=[blocks_outward(99, "PB-99")]),
    ]
    patch_pager(monkeypatch, issues, pages_fetched=3)

    snapshot = jira_backend.refresh_board_snapshot(1)
    by_id = {task.task_id: task for task in snapshot.tasks}

    assert by_id[10].blocking == [20]
    assert by_id[20].blocked_by == [10]

    unresolved = by_id[30].metadata["unresolved_links"]
    assert unresolved == [
        {
            "id": "99",
            "key": "PB-99",
            "relation": "blocks",
            "direction": "outward",
        }
    ]
    assert snapshot.unresolved_link_count == 1


def test_get_board_infos_does_not_hit_network(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    pager = patch_pager(monkeypatch, [make_issue(10, "PB-1")])

    infos = jira_backend.get_board_infos()
    assert len(infos) == 1
    assert infos[0]["amount_tasks"] is None
    assert infos[0]["stale"] is True
    pager.assert_not_called()

    jira_backend.refresh_board_snapshot(1)
    infos = jira_backend.get_board_infos()
    assert infos[0]["amount_tasks"] == 1
    assert infos[0]["stale"] is False


def test_get_tasks_by_ids_uses_single_batched_jql(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_pager(monkeypatch, [make_issue(10, "PB-1")])
    jira_backend.refresh_board_snapshot(1)

    paged = MagicMock(
        return_value=PagedSearchResult(
            issues=(make_issue(777, "PB-777"),),
            total=1,
            pages_fetched=1,
            page_size=100,
            watermark=None,
            is_cloud=False,
        )
    )
    monkeypatch.setattr(backend_module, "search_issues_paged", paged)

    tasks = jira_backend.get_tasks_by_ids([10, 777])

    assert [task.task_id for task in tasks] == [10, 777]
    paged.assert_called_once()
    jql_arg = paged.call_args.args[1]
    assert jql_arg == 'id in ("777")'


def test_update_task_status_conflicts_on_newer_remote_update(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_pager(monkeypatch, [make_issue(10, "PB-1")])
    snapshot = jira_backend.refresh_board_snapshot(1)

    state_mock = MagicMock(
        return_value={
            "key": "PB-1",
            "status": "In Progress",
            "updated": datetime(2026, 2, 1, tzinfo=UTC),
        }
    )
    transitions_mock = MagicMock()
    monkeypatch.setattr(backend_module, "get_issue_state", state_mock)
    monkeypatch.setattr(backend_module, "get_transitions", transitions_mock)

    moved = snapshot.tasks[0].model_copy(update={"column": 2})
    result = jira_backend.update_task_status(moved)

    assert result["success"] is False
    assert result["conflict"] is True
    state_mock.assert_called_once()
    transitions_mock.assert_not_called()


def test_update_task_status_succeeds_with_watermark(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_pager(monkeypatch, [make_issue(10, "PB-1")])
    snapshot = jira_backend.refresh_board_snapshot(1)

    monkeypatch.setattr(
        backend_module,
        "get_issue_state",
        MagicMock(
            return_value={
                "key": "PB-1",
                "status": "To Do",
                "updated": WATERMARK,
            }
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "get_transitions",
        MagicMock(return_value=[{"id": 5, "to": "In Progress"}]),
    )
    transition_mock = MagicMock(
        return_value={"success": True, "message": "Transitioned PB-1"}
    )
    monkeypatch.setattr(backend_module, "set_issue_status", transition_mock)

    moved = snapshot.tasks[0].model_copy(update={"column": 2})
    result = jira_backend.update_task_status(moved)

    assert result["success"] is True
    assert result["watermark"] == WATERMARK.isoformat()
    transition_mock.assert_called_once()


def test_delete_board_removes_config_entry_and_snapshot(
    jira_backend: JiraBackend,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_pager(monkeypatch, [make_issue(10, "PB-1")])
    jira_backend.refresh_board_snapshot(1)
    cache_file = jira_backend._cache_file(1)
    assert cache_file.exists()

    jira_backend.delete_board(1)

    assert jira_backend.settings.jqls == []
    assert not cache_file.exists()
    assert jira_backend.get_cached_snapshot(1) is None
