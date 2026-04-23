"""Shared fixtures: synthetic GitHub API payloads and classifications.

Every test should rely on these; no test should hit a live API.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from agentfail.schema import Classification, RawIssue


@pytest.fixture
def github_issue_payload() -> dict[str, Any]:
    """A realistic payload shaped like what /repos/:owner/:repo/issues returns."""
    return {
        "number": 1234,
        "node_id": "I_kwDOAB12_5M6F7g8",
        "title": "Agent loops forever when tool raises on retry",
        "body": (
            "Running CrewAI 0.x, the planner agent gets stuck in an infinite "
            "tool-call loop when the search tool raises a 429 twice in a row. "
            "Traceback shows `TaskExecutionError` but the outer loop retries "
            "without backoff. Expected: finite retries with exponential backoff."
        ),
        "html_url": "https://github.com/example/repo/issues/1234",
        "labels": [{"name": "type:bug"}, {"name": "priority:high"}],
        "state": "open",
        "author_association": "CONTRIBUTOR",
        "created_at": "2026-03-10T09:30:00Z",
        "updated_at": "2026-03-12T18:15:00Z",
        "closed_at": None,
        "comments": 4,
    }


@pytest.fixture
def feature_request_payload() -> dict[str, Any]:
    """A non-failure payload that the heuristic should drop."""
    return {
        "number": 5678,
        "node_id": "I_kwDOAB12_5M6F7g9",
        "title": "Add support for streaming tool responses",
        "body": "It would be great to...",
        "html_url": "https://github.com/example/repo/issues/5678",
        "labels": [{"name": "enhancement"}],
        "state": "open",
        "created_at": "2026-04-01T09:30:00Z",
        "updated_at": "2026-04-01T09:30:00Z",
        "closed_at": None,
        "comments": 0,
    }


@pytest.fixture
def pr_payload() -> dict[str, Any]:
    """A pull-request entry from the same endpoint."""
    return {
        "number": 9999,
        "node_id": "PR_kwDOAB12_5M6F7g0",
        "title": "Fix infinite loop",
        "body": "This patch...",
        "html_url": "https://github.com/example/repo/pull/9999",
        "labels": [],
        "state": "closed",
        "created_at": "2026-03-15T09:30:00Z",
        "updated_at": "2026-03-16T09:30:00Z",
        "closed_at": "2026-03-16T10:00:00Z",
        "comments": 1,
        "pull_request": {"url": "https://api.github.com/..."},
    }


@pytest.fixture
def raw_issue() -> RawIssue:
    return RawIssue(
        framework_slug="crewai",
        issue_number=1234,
        node_id="I_kwDOAB12_5M6F7g8",
        title="Agent loops forever when tool raises on retry",
        body="CrewAI stuck in loop on repeated 429...",
        url="https://github.com/example/repo/issues/1234",
        labels=("type:bug",),
        state="open",
        is_pull_request=False,
        created_at=datetime(2026, 3, 10, 9, 30, tzinfo=UTC),
        updated_at=datetime(2026, 3, 12, 18, 15, tzinfo=UTC),
        closed_at=None,
        comment_count=4,
    )


@pytest.fixture
def classification_confident() -> Classification:
    return Classification(
        locus="agent",
        phase="action",
        symptom="loop",
        root_cause="api_misuse",
        confidence=0.92,
        reasoning="Reported repeated tool-call loop without backoff; framework orchestration at fault.",
        needs_review=False,
    )


@pytest.fixture
def classification_low_confidence() -> Classification:
    return Classification(
        locus="unknown",
        phase="unknown",
        symptom="unknown",
        root_cause="unknown",
        confidence=0.3,
        reasoning="Too vague to classify.",
        needs_review=True,
    )
