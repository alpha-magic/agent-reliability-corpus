"""Typed data boundaries for the agentfail corpus.

Every IO surface uses these models. They also double as the column schema
for the published HF dataset — keep field names stable across releases
(use deprecation, don't rename).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# --- Taxonomy axes ---------------------------------------------------------
# Four orthogonal axes derived from the unified taxonomy (see taxonomy.py).
# Each axis is a closed Literal so the classifier's tool-use schema is strict.

Locus = Literal[
    "model",  # LLM itself (hallucination, wrong tool choice, token limit)
    "agent",  # single-agent reasoning/memory (forgetting, loop, misinterpretation)
    "framework",  # framework code (API misuse, version mismatch, doc desync)
    "workflow",  # multi-step/multi-agent orchestration (deadlock, role confusion)
    "platform",  # runtime/infra (timeout, cost overrun, env issue)
    "unknown",
]

Phase = Literal[
    "planning",
    "action",  # tool use, environment interaction
    "reflection",
    "memory",
    "coordination",  # multi-agent only
    "infra",
    "unknown",
]

Symptom = Literal[
    "wrong_output",
    "no_output",  # stuck, timeout, hang
    "cost_overrun",
    "crash",  # exception, traceback
    "security",  # safety/security violation, prompt injection
    "loop",  # non-termination
    "unknown",
]

RootCause = Literal[
    "api_misuse",
    "api_incompatibility",  # version mismatch, breaking change
    "doc_desync",
    "model_limitation",
    "prompt_issue",
    "data_quality",
    "infrastructure",
    "unknown",
]

ClassifierTier = Literal["haiku", "sonnet", "opus", "human"]


# --- Core models -----------------------------------------------------------


class Framework(BaseModel):
    """A framework we scrape issues from."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(..., description="Short ID, e.g. 'langchain', used in URLs and filenames")
    repo: str = Field(..., description="GitHub owner/repo, e.g. 'langchain-ai/langchain'")
    display_name: str
    homepage: HttpUrl | None = None


class RawIssue(BaseModel):
    """A GitHub issue scraped from a framework repository, pre-classification.

    We store the minimum needed to classify and cite; keep title/body verbatim
    so later reclassification (e.g. with a better classifier) is reproducible.
    """

    model_config = ConfigDict(frozen=True)

    # Identifiers
    framework_slug: str
    issue_number: int
    node_id: str  # GitHub's stable global ID

    # Content (verbatim at scrape time)
    title: str
    body: str | None = None
    url: HttpUrl
    labels: tuple[str, ...] = ()

    # State
    state: Literal["open", "closed"]
    author_association: str | None = None
    is_pull_request: bool = False

    # Timestamps
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None

    # Resolution signal
    comment_count: int = 0


class Classification(BaseModel):
    """Structured output from the classifier — one row per issue.

    The classifier returns this object directly; Pydantic AI enforces
    the schema via tool-use. `reasoning` is kept short (~1 sentence) to
    control cost, and is for auditability, not for downstream training.
    """

    locus: Locus
    phase: Phase
    symptom: Symptom
    root_cause: RootCause
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=500)
    needs_review: bool = Field(
        default=False,
        description="Classifier flagged this as not cleanly fitting the taxonomy.",
    )


class ClassifiedIssue(BaseModel):
    """A RawIssue + its classification + reproducibility metadata.

    This is the schema of the primary `issues` config on the HF dataset.
    """

    # Issue fields (flattened from RawIssue for clean column access)
    framework_slug: str
    issue_number: int
    node_id: str
    title: str
    body: str | None
    url: HttpUrl
    labels: tuple[str, ...]
    state: Literal["open", "closed"]
    is_pull_request: bool
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    comment_count: int

    # Classification fields (flattened from Classification)
    locus: Locus
    phase: Phase
    symptom: Symptom
    root_cause: RootCause
    confidence: float
    reasoning: str
    needs_review: bool

    # Reproducibility
    classifier_tier: ClassifierTier
    classifier_model: str = Field(
        ..., description="Pinned model ID, e.g. 'claude-haiku-4-5-20251001'"
    )
    classifier_version: str = Field(..., description="agentfail release that produced this label")
    classified_at: datetime

    @classmethod
    def from_parts(
        cls,
        issue: RawIssue,
        classification: Classification,
        *,
        classifier_tier: ClassifierTier,
        classifier_model: str,
        classifier_version: str,
        classified_at: datetime,
    ) -> ClassifiedIssue:
        return cls(
            framework_slug=issue.framework_slug,
            issue_number=issue.issue_number,
            node_id=issue.node_id,
            title=issue.title,
            body=issue.body,
            url=issue.url,
            labels=issue.labels,
            state=issue.state,
            is_pull_request=issue.is_pull_request,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            closed_at=issue.closed_at,
            comment_count=issue.comment_count,
            locus=classification.locus,
            phase=classification.phase,
            symptom=classification.symptom,
            root_cause=classification.root_cause,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            needs_review=classification.needs_review,
            classifier_tier=classifier_tier,
            classifier_model=classifier_model,
            classifier_version=classifier_version,
            classified_at=classified_at,
        )


class CrossLink(BaseModel):
    """A link from one of our issues to a record in an academic failure corpus.

    `corpus_id` identifies which academic dataset (e.g. 'mast', 'agentfail');
    `external_id` is the record ID within that dataset. `similarity` and
    `method` let downstream users filter by link quality.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str  # matches ClassifiedIssue.node_id
    corpus_id: Literal["mast", "agentfail", "agent_error", "faults_agentic", "framework_bugs"]
    external_id: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    method: Literal["exact_match", "semantic", "manual"]


# --- Reference data --------------------------------------------------------


class TaxonomyLabel(BaseModel):
    """One row of the `taxonomy` config on the published dataset.

    Flattens the 4-axis space into a reference table with definitions and
    citations to the academic prior work each label derives from.
    """

    model_config = ConfigDict(frozen=True)

    axis: Literal["locus", "phase", "symptom", "root_cause"]
    label: str
    definition: str
    derived_from: tuple[str, ...] = Field(
        default=(),
        description="Citation keys for papers this label traces back to.",
    )
