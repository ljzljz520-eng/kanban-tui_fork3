from collections.abc import Iterable
from typing import TYPE_CHECKING

from kanban_tui.config import Backends
from kanban_tui.modal.modal_auth_screen import ModalAuthScreen
from kanban_tui.modal.modal_jira_url_screen import ModalBaseUrlScreen

if TYPE_CHECKING:
    from kanban_tui.app import KanbanTui

from rich.text import Text
from textual import on, work
from textual.events import ScreenResume
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Header
from textual.worker import get_current_worker

from kanban_tui.classes.board import Board
from kanban_tui.refresh_state import RefreshPhase
from kanban_tui.widgets.board_widgets import KanbanBoard
from kanban_tui.widgets.custom_widgets import KanbanTuiFooter


class BoardScreen(Screen):
    app: "KanbanTui"
    active_board: reactive[Board | None] = reactive(None, init=False)

    def compose(self) -> Iterable[Widget]:
        yield KanbanBoard()
        yield Header()
        yield KanbanTuiFooter()

    def on_mount(self) -> None:
        self.watch(self.app, "refresh_state", self.watch_refresh_state, init=False)
        self.watch_refresh_state(self.app.refresh_state)

    def watch_active_board(self):
        if self.active_board:
            border_title = Text.from_markup(
                f" [red]Active Board:[/] {self.active_board.full_name}"
            )
            self.query_one(KanbanBoard).border_title = border_title

    def watch_refresh_state(self, state) -> None:
        """Render the loading/stale/error status on the board border."""
        board = self.query_one_optional(KanbanBoard)
        if board is None:
            return

        if state.phase == RefreshPhase.LOADING:
            board.border_subtitle = Text.from_markup(
                "[yellow]\u27f3 Loading issues\u2026[/]"
            )
        elif state.phase == RefreshPhase.STALE:
            board.border_subtitle = Text.from_markup(
                f"[yellow]\u26a0 Stale[/] [grey50]{state.message}[/]"
            )
        elif state.phase == RefreshPhase.ERROR:
            board.border_subtitle = Text.from_markup(
                f"[red]\u2717 {state.message}[/]"
            )
        else:
            board.border_subtitle = None

    async def ensure_active_board(self):
        if not self.active_board:
            await self.query_one(KanbanBoard).action_show_boards()

    async def ensure_api_key(self):
        if not self.app.backend.api_key:
            await self.app.push_screen_wait(ModalAuthScreen())

    async def ensure_base_url(self):
        if not self.app.backend.settings.base_url:
            await self.app.push_screen_wait(ModalBaseUrlScreen())

    @work(group="board-refresh", exclusive=True)
    @on(ScreenResume)
    async def load_kanban_board(self, event: ScreenResume | None = None):
        self.set_reactive(BoardScreen.active_board, self.app.active_board)

        match self.app.config.backend.mode:
            case Backends.JIRA:
                await self.ensure_api_key()
                if not self.app.backend.api_key:
                    worker = get_current_worker()
                    worker.cancel()
                    self.app.config.set_backend(Backends("sqlite"))
                    self.app.exit(return_code=1, message="Please enter a valid api key")

                await self.ensure_base_url()
                if not self.app.backend.settings.base_url:
                    worker = get_current_worker()
                    worker.cancel()
                    self.app.config.set_backend(Backends("sqlite"))
                    self.app.exit(
                        return_code=1, message="Please enter a valid jira base url"
                    )

        await self.ensure_active_board()

        if self.app.config.backend.mode != Backends.JIRA:
            if self.app.needs_refresh:
                self.app.update_task_list()
                await self.query_one(KanbanBoard).refresh_columns()
                self.app.needs_refresh = False
            return

        if self.app.active_board is None:
            return
        await self.load_jira_board(self.app.active_board.board_id)

    async def load_jira_board(self, board_id: int) -> None:
        """Generation-guarded Jira refresh.

        Slow or failing refreshes never wipe a rendered board; stale
        generations (newer refresh started / board switched) are discarded
        without writing back.
        """
        generation = self.app.begin_refresh(board_id)

        try:
            snapshot = await self.app.backend.arefresh_board(board_id)
        except Exception as exc:  # surface any Jira failure as state
            if self.app.is_refresh_current(generation, board_id):
                fallback = self.app.backend.get_cached_snapshot(board_id)
                message = str(exc)
                if fallback is not None:
                    self.app.task_list = list(fallback.tasks)
                    await self.query_one(KanbanBoard).refresh_columns()
                    if self.app.settle_refresh(
                        generation,
                        board_id,
                        RefreshPhase.STALE,
                        message="showing cached data, refresh failed",
                    ):
                        self.app.notify(
                            title="Jira refresh failed - showing cached data",
                            message=message,
                            severity="warning",
                            timeout=8,
                        )
                elif self.app.settle_refresh(
                    generation, board_id, RefreshPhase.ERROR, message=message
                ):
                    self.app.notify(
                        title="Jira refresh failed",
                        message=message,
                        severity="error",
                        timeout=8,
                    )
                self.app.needs_refresh = False
            return

        # A newer refresh started or the user switched boards meanwhile:
        # this snapshot belongs to a stale generation and must not be shown.
        if not self.app.is_refresh_current(generation, board_id):
            return

        self.app.task_list = list(snapshot.tasks)
        await self.query_one(KanbanBoard).refresh_columns()
        self.app.settle_refresh(generation, board_id, RefreshPhase.IDLE)
        self.app.needs_refresh = False
