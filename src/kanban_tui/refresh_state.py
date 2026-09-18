"""Generation-tagged refresh state shared by the app and board screen.

Kept in its own module to avoid the app <-> board_screen import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RefreshPhase(StrEnum):
    """Lifecycle of an asynchronous (Jira) board refresh."""

    IDLE = "idle"
    LOADING = "loading"
    STALE = "stale"
    ERROR = "error"


@dataclass(frozen=True)
class RefreshState:
    """Status of a board refresh tagged with its generation.

    Only the worker owning the newest generation for the active board may
    publish its result or transition the state; older refreshes (slow
    network, auto refresh, board switched mid-flight) are discarded.
    """

    phase: RefreshPhase = RefreshPhase.IDLE
    message: str = ""
    board_id: int | None = None
    generation: int = 0
