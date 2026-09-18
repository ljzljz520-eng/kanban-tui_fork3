from kanban_tui.app import KanbanTui
from kanban_tui.refresh_state import RefreshPhase


def test_begin_refresh_bumps_generation_and_sets_loading(no_task_app: KanbanTui):
    board = no_task_app.backend.get_boards()[0]
    no_task_app.active_board = board

    generation = no_task_app.begin_refresh(board.board_id)

    assert generation == 1
    assert no_task_app.refresh_state.phase == RefreshPhase.LOADING
    assert no_task_app.refresh_state.board_id == board.board_id
    assert no_task_app.refresh_state.generation == 1
    assert no_task_app.begin_refresh(board.board_id) == 2


def test_stale_generation_rejected_after_board_switch(no_task_app: KanbanTui):
    first = no_task_app.backend.get_boards()[0]
    second_board = no_task_app.backend.create_new_board(
        name="Second", icon=":mag:"
    )

    no_task_app.active_board = first
    generation = no_task_app.begin_refresh(first.board_id)

    # User switches boards while the refresh is in flight
    no_task_app.active_board = second_board

    assert no_task_app.is_refresh_current(generation, first.board_id) is False
    # Old worker must not be allowed to publish state for the old board
    assert (
        no_task_app.settle_refresh(
            generation, first.board_id, RefreshPhase.IDLE
        )
        is False
    )
    # State stays owned by the in-flight generation for the *new* board's
    # refresh; the stale worker changed nothing.
    assert no_task_app.refresh_state.generation == generation
    assert no_task_app.refresh_state.board_id == first.board_id


def test_newer_generation_wins_and_settles(no_task_app: KanbanTui):
    board = no_task_app.backend.get_boards()[0]
    no_task_app.active_board = board

    first_generation = no_task_app.begin_refresh(board.board_id)
    second_generation = no_task_app.begin_refresh(board.board_id)

    assert no_task_app.is_refresh_current(first_generation, board.board_id) is False
    assert no_task_app.is_refresh_current(second_generation, board.board_id) is True

    assert (
        no_task_app.settle_refresh(
            first_generation, board.board_id, RefreshPhase.ERROR, message="late"
        )
        is False
    )
    assert no_task_app.refresh_state.phase == RefreshPhase.LOADING

    assert (
        no_task_app.settle_refresh(
            second_generation, board.board_id, RefreshPhase.IDLE
        )
        is True
    )
    assert no_task_app.refresh_state.phase == RefreshPhase.IDLE
