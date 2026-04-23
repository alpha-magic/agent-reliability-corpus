"""Tiered agent-failure classifier.

Agent A's workhorse. Given a `RawIssue`, returns a `ClassifiedIssue` by
routing through Haiku → Sonnet → Opus based on the classifier's own
confidence. Uses the Anthropic SDK directly for fine-grained control over
prompt caching and tool-use enforcement — Pydantic AI is reserved for
Agent B where its multi-step agent loop earns its keep.

Reproducibility choices:
- Pinned model IDs (no aliases)
- temperature=0
- Tool-use with strict `Classification` schema
- All four versioned fields recorded on the output row

Cost control:
- The taxonomy block (~3.6KB, stable across runs) is cache_control-marked,
  so it hits the Anthropic prompt cache after the first call (~90% savings
  on the cached portion).
- Only the issue text varies per call — small, un-cacheable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam

from agentfail import __version__
from agentfail.schema import Classification, ClassifiedIssue, ClassifierTier, RawIssue
from agentfail.taxonomy import render_taxonomy_for_prompt

log = structlog.get_logger(__name__)

# --- Pinned model IDs -----------------------------------------------------
# Bumped deliberately, reflected in `classifier_model` on every row.

MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6-20260101"  # placeholder if pinning a dated variant
MODEL_OPUS = "claude-opus-4-7"

# --- Escalation thresholds ------------------------------------------------

HAIKU_ACCEPT_THRESHOLD = 0.80  # below this, escalate to Sonnet
SONNET_ACCEPT_THRESHOLD = 0.70  # below this, escalate to Opus
# Opus output is always accepted; if its confidence is still low the row is
# marked needs_review=True for human audit.

# --- Tool definition (shared across tiers) --------------------------------

_TOOL_NAME = "classify_failure"

# Derive the tool's input schema from the Pydantic model so any schema change
# flows through automatically. `additionalProperties=False` keeps the
# classifier from inventing fields.
_CLASSIFICATION_SCHEMA: dict[str, Any] = Classification.model_json_schema()
_CLASSIFICATION_SCHEMA["additionalProperties"] = False

_TOOL: ToolParam = cast(
    ToolParam,
    {
        "name": _TOOL_NAME,
        "description": (
            "Return your classification of the GitHub issue against the "
            "unified agent-failure taxonomy."
        ),
        "input_schema": _CLASSIFICATION_SCHEMA,
    },
)

_TOOL_CHOICE: ToolChoiceToolParam = cast(
    ToolChoiceToolParam,
    {"type": "tool", "name": _TOOL_NAME},
)


@dataclass(frozen=True)
class ClassifierResult:
    classification: Classification
    tier: ClassifierTier
    model_id: str


class Classifier:
    """Tiered classifier. Constructed once per pipeline run; caches the
    Anthropic client and the rendered taxonomy block."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        haiku_model: str = MODEL_HAIKU,
        sonnet_model: str = MODEL_SONNET,
        opus_model: str = MODEL_OPUS,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required. Set it in the environment or "
                "pass api_key=... to Classifier()."
            )
        self._client = Anthropic(api_key=key)
        self._taxonomy_prompt = render_taxonomy_for_prompt()
        self._models = {
            "haiku": haiku_model,
            "sonnet": sonnet_model,
            "opus": opus_model,
        }

    # --- One-shot per tier ----------------------------------------------

    def _call(self, tier: ClassifierTier, issue: RawIssue) -> Classification:
        assert tier in ("haiku", "sonnet", "opus")
        model_id = self._models[tier]

        # System prompt is a two-block structure: taxonomy (cached) + static
        # directive. Keeping the cached block first maximizes prefix-match hits.
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": self._taxonomy_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    "You are a careful, conservative classifier. When in doubt, "
                    "prefer `unknown` and set `needs_review=true`. Do not invent "
                    "labels outside the taxonomy."
                ),
            },
        ]

        user_message: MessageParam = {
            "role": "user",
            "content": _render_issue_for_classification(issue),
        }

        response = self._client.messages.create(
            model=model_id,
            max_tokens=1024,
            temperature=0,
            system=cast(Any, system_blocks),
            tools=[_TOOL],
            tool_choice=_TOOL_CHOICE,
            messages=[user_message],
        )

        # Find the tool_use block; parse its input into our Pydantic model.
        for block in response.content:
            if block.type == "tool_use" and block.name == _TOOL_NAME:
                return Classification.model_validate(block.input)

        raise RuntimeError(
            f"Classifier response from {model_id} did not contain a "
            f"{_TOOL_NAME} tool call. Content: {response.content!r}"
        )

    # --- Tier escalation ------------------------------------------------

    def classify(self, issue: RawIssue) -> ClassifierResult:
        """Run the tier ladder; return the first confident result."""
        haiku_out = self._call("haiku", issue)
        if haiku_out.confidence >= HAIKU_ACCEPT_THRESHOLD and not haiku_out.needs_review:
            log.debug("classify.tier_accept", tier="haiku", node_id=issue.node_id)
            return ClassifierResult(haiku_out, "haiku", self._models["haiku"])

        log.info(
            "classify.escalate",
            from_tier="haiku",
            to_tier="sonnet",
            node_id=issue.node_id,
            confidence=haiku_out.confidence,
            needs_review=haiku_out.needs_review,
        )
        sonnet_out = self._call("sonnet", issue)
        if sonnet_out.confidence >= SONNET_ACCEPT_THRESHOLD and not sonnet_out.needs_review:
            return ClassifierResult(sonnet_out, "sonnet", self._models["sonnet"])

        log.info(
            "classify.escalate",
            from_tier="sonnet",
            to_tier="opus",
            node_id=issue.node_id,
            confidence=sonnet_out.confidence,
        )
        opus_out = self._call("opus", issue)
        return ClassifierResult(opus_out, "opus", self._models["opus"])

    # --- Convenience: classify + wrap into ClassifiedIssue row ---------

    def classify_to_row(self, issue: RawIssue) -> ClassifiedIssue:
        result = self.classify(issue)
        return ClassifiedIssue.from_parts(
            issue,
            result.classification,
            classifier_tier=result.tier,
            classifier_model=result.model_id,
            classifier_version=__version__,
            classified_at=datetime.now(UTC),
        )


# --- Helpers --------------------------------------------------------------

_MAX_BODY_CHARS = 4000  # truncate to keep input tokens bounded


def _render_issue_for_classification(issue: RawIssue) -> str:
    """Render a RawIssue into the prompt payload. Deterministic.

    Kept terse to bound input-token cost; the classifier does not need the
    full comment thread, only the opening report.
    """
    body = (issue.body or "").strip()
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + "\n\n[truncated]"

    labels_str = ", ".join(issue.labels) if issue.labels else "(none)"

    return (
        f"Framework: {issue.framework_slug}\n"
        f"Issue #{issue.issue_number} ({issue.state})\n"
        f"Labels: {labels_str}\n"
        f"URL: {issue.url}\n\n"
        f"Title: {issue.title}\n\n"
        f"Body:\n{body or '(empty)'}"
    )
