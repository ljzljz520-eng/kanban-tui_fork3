from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kanban_tui.classes.task import Task


class JiraUser(BaseModel):
    """Jira user model"""

    accountId: str
    displayName: str
    emailAddress: str | None = None
    active: bool = True


class JiraStatus(BaseModel):
    """Jira status model"""

    id: str
    name: str
    statusCategory: dict[str, Any] = Field(default_factory=dict)


class JiraIssueType(BaseModel):
    """Jira issue type model"""

    id: str
    name: str
    subtask: bool = False


class JiraPriority(BaseModel):
    """Jira priority model"""

    id: str
    name: str


class JiraIssueLink(BaseModel):
    """Jira issue link model"""

    id: str
    type: dict[str, Any]
    outwardIssue: dict[str, Any] | None = None
    inwardIssue: dict[str, Any] | None = None


class JiraIssue(BaseModel):
    """Jira issue model with helper properties for accessing nested fields"""

    key: str
    id: str
    fields: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Get issue summary/title"""
        return self.fields.get("summary", "")

    @property
    def description(self) -> str:
        """Get issue description"""
        desc = self.fields.get("description")
        if desc is None:
            return ""
        # Handle both string and complex description formats
        if isinstance(desc, str):
            return desc
        # For Atlassian Document Format (ADF), extract text content
        if isinstance(desc, dict):
            return self._extract_text_from_adf(desc)
        return str(desc)

    def _extract_text_from_adf(self, adf: dict) -> str:
        """Extract plain text from Atlassian Document Format"""
        if not isinstance(adf, dict):
            return ""

        text_parts = []

        def extract_text(node):
            if isinstance(node, dict):
                # Handle text nodes
                if node.get("type") == "text":
                    text_parts.append(node.get("text", ""))
                # Recursively process content
                if "content" in node:
                    for child in node["content"]:
                        extract_text(child)
            elif isinstance(node, list):
                for item in node:
                    extract_text(item)

        extract_text(adf)
        return " ".join(text_parts)

    @property
    def assignee(self) -> str | None:
        """Get assignee display name"""
        assignee = self.fields.get("assignee")
        if not assignee:
            return None
        return assignee.get("displayName") or assignee.get("name")

    @property
    def assignee_email(self) -> str | None:
        """Get assignee email"""
        assignee = self.fields.get("assignee")
        if not assignee:
            return None
        return assignee.get("emailAddress")

    @property
    def reporter(self) -> str | None:
        """Get reporter display name"""
        reporter = self.fields.get("reporter")
        if not reporter:
            return None
        return reporter.get("displayName") or reporter.get("name")

    @property
    def priority(self) -> str | None:
        """Get priority name"""
        priority = self.fields.get("priority")
        if not priority:
            return None
        return priority.get("name")

    @property
    def issue_type(self) -> str:
        """Get issue type name"""
        issuetype = self.fields.get("issuetype", {})
        return issuetype.get("name", "Task")

    @property
    def status(self) -> str:
        """Get status name"""
        status = self.fields.get("status", {})
        return status.get("name", "To Do")

    @property
    def status_category(self) -> str:
        """Get status category (To Do, In Progress, Done)"""
        status = self.fields.get("status", {})
        category = status.get("statusCategory", {})
        return category.get("name", "To Do")

    @property
    def created(self) -> datetime:
        """Get creation timestamp"""
        created_str = self.fields.get("created", "")
        if not created_str:
            return datetime.now()
        # Jira returns ISO format with timezone
        return datetime.fromisoformat(created_str)

    @property
    def updated(self) -> datetime | None:
        """Get last update timestamp"""
        updated_str = self.fields.get("updated")
        if not updated_str:
            return None
        return datetime.fromisoformat(updated_str)

    @property
    def due_date(self) -> datetime | None:
        """Get due date"""
        due = self.fields.get("duedate")
        if not due:
            return None
        # Due date is just YYYY-MM-DD format
        return datetime.fromisoformat(due)

    @property
    def labels(self) -> list[str]:
        """Get labels/tags"""
        return self.fields.get("labels", [])

    @property
    def components(self) -> list[str]:
        """Get component names"""
        components = self.fields.get("components", [])
        return [comp.get("name", "") for comp in components if isinstance(comp, dict)]

    @property
    def issue_links(self) -> list[dict[str, Any]]:
        """Get issue links"""
        return self.fields.get("issuelinks", [])

    @property
    def resolution(self) -> str | None:
        """Get resolution name"""
        resolution = self.fields.get("resolution")
        if not resolution:
            return None
        return resolution.get("name")

    @property
    def resolution_date(self) -> datetime | None:
        """Get resolution date (when issue was resolved/closed)"""
        resolved_str = self.fields.get("resolutiondate")
        if not resolved_str:
            return None
        return datetime.fromisoformat(resolved_str)


class JiraProject(BaseModel):
    """Jira project model"""

    key: str
    name: str
    id: str
    projectTypeKey: str = "software"


@dataclass(frozen=True)
class JiraBoardSnapshot:
    """Immutable, fully verified result set for one board (JQL + mapping).

    A snapshot is published only after *all* pages have been collected and
    ``total`` has been verified, so consumers never see partially fetched
    boards. Dependency resolution runs on the complete issue set.

    Attributes:
        board_id: Virtual board id (the JQL entry id).
        jql: JQL the issues were fetched with (identity/staleness check).
        column_mapping: Status-to-column mapping used for this snapshot.
        tasks: Converted, dependency-resolved tasks.
        total: Server-reported total issue count (``None`` on Cloud cursor
            pagination).
        pages_fetched: Number of pages consumed from Jira.
        watermark: Newest remote ``updated`` timestamp across the issues.
        fetched_at: Local publication time.
        unresolved_link_count: Issue links pointing outside the query scope.
    """

    board_id: int
    jql: str
    column_mapping: Mapping[str, int]
    tasks: tuple[Task, ...]
    total: int | None
    pages_fetched: int
    watermark: datetime | None
    fetched_at: datetime
    unresolved_link_count: int = 0

    @property
    def is_complete(self) -> bool:
        return self.total is None or len(self.tasks) == self.total
