"""Classifier tests — stub the Pydantic AI agent so no network is hit.

The real Classifier wraps a Pydantic AI `Agent` whose `run_sync()`
returns a result object with `.output` (the Classification) and a
`.usage()` callable. Tests substitute that with a stub so we can
control what the model "returns" without going through the real
provider, while still exercising the Classifier's public surface
(`classify`, `classify_to_row`, ClassifierResult shaping, usage
logging).

Provider-specific quirks (max_tokens vs max_completion_tokens, the
Mistral type=None tool-call thing, OpenAI's temperature constraints
on reasoning models) are deliberately NOT unit-tested here — those
are Pydantic AI's responsibility now and would require integration
tests against real provider APIs to verify meaningfully. They are
exercised by the `arc-relabel` smoke runs against each target
provider documented in PAPER.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentfail.classify import (
    MODEL_V4_PRO,
    Classifier,
    _render_issue_for_classification,
)
from agentfail.schema import Classification

# --- Stub agent ---------------------------------------------------------


@dataclass
class _StubUsage:
    input_tokens: int = 1000
    output_tokens: int = 200
    total_tokens: int = 1200
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubResult:
    output: Classification
    _usage: _StubUsage = field(default_factory=_StubUsage)

    def usage(self) -> _StubUsage:
        return self._usage


@dataclass
class _StubAgent:
    """Stand-in for `pydantic_ai.Agent` — captures prompts and replays
    pre-canned Classification outputs."""

    outputs: list[Classification]
    calls: list[str] = field(default_factory=list)

    def run_sync(self, prompt: str) -> _StubResult:
        if not self.outputs:
            raise AssertionError("StubAgent ran out of pre-canned outputs")
        self.calls.append(prompt)
        return _StubResult(output=self.outputs.pop(0))


def _classifier_with_outputs(
    outputs: list[Classification], model: str = MODEL_V4_PRO
) -> tuple[Classifier, _StubAgent]:
    """Construct a Classifier with the agent replaced by a stub."""
    c = Classifier.__new__(Classifier)  # bypass __init__ (needs API key)
    stub = _StubAgent(outputs=list(outputs))
    c._agent = stub  # type: ignore[attr-defined]
    c._model = model  # type: ignore[attr-defined]
    return c, stub


def _make_classification(confidence: float, needs_review: bool = False) -> Classification:
    return Classification(
        locus="agent",
        phase="action",
        symptom="loop",
        root_cause="api_misuse",
        confidence=confidence,
        reasoning="test",
        needs_review=needs_review,
    )


# --- Behavior tests -----------------------------------------------------


def test_classify_returns_v4_pro_tier(raw_issue):
    c, stub = _classifier_with_outputs([_make_classification(confidence=0.9)])
    result = c.classify(raw_issue)
    assert result.tier == "v4_pro"
    assert result.model_id == MODEL_V4_PRO
    assert result.classification.confidence == 0.9
    assert len(stub.calls) == 1


def test_classify_propagates_needs_review(raw_issue):
    """needs_review propagates straight to the output row regardless of
    confidence — there is no escalation tier to gate it on."""
    c, _ = _classifier_with_outputs([_make_classification(confidence=0.95, needs_review=True)])
    result = c.classify(raw_issue)
    assert result.classification.needs_review is True


def test_classify_accepts_low_confidence_output(raw_issue):
    """Single-tier: low-confidence outputs are still accepted (no fallback).
    Downstream filters can use confidence + needs_review to drop them."""
    c, _ = _classifier_with_outputs([_make_classification(confidence=0.2, needs_review=True)])
    result = c.classify(raw_issue)
    assert result.tier == "v4_pro"
    assert result.classification.confidence == 0.2


def test_classify_to_row_records_reproducibility_metadata(raw_issue):
    """The wrapped row has the reproducibility metadata the paper relies on."""
    c, _ = _classifier_with_outputs([_make_classification(confidence=0.9)])
    row = c.classify_to_row(raw_issue)
    assert row.classifier_tier == "v4_pro"
    assert row.classifier_model == MODEL_V4_PRO
    assert row.classifier_version  # set from agentfail.__version__
    assert row.classified_at is not None


def test_classify_passes_rendered_issue_to_agent(raw_issue):
    """The agent receives the rendered prompt, not raw issue fields —
    bug-prevention against accidentally passing a JSON dump or similar."""
    c, stub = _classifier_with_outputs([_make_classification(confidence=0.9)])
    c.classify(raw_issue)
    assert len(stub.calls) == 1
    sent = stub.calls[0]
    assert raw_issue.framework_slug in sent
    assert raw_issue.title in sent
    assert str(raw_issue.issue_number) in sent


# --- Renderer tests -----------------------------------------------------


def test_render_issue_includes_key_fields(raw_issue):
    rendered = _render_issue_for_classification(raw_issue)
    assert raw_issue.framework_slug in rendered
    assert str(raw_issue.issue_number) in rendered
    assert raw_issue.title in rendered
    assert raw_issue.state in rendered


def test_render_issue_truncates_long_body(raw_issue):
    long_body = "x" * 5000
    raw = raw_issue.model_copy(update={"body": long_body})
    rendered = _render_issue_for_classification(raw)
    assert "[truncated]" in rendered
    # No raw 5000-char block survives
    assert "x" * 5000 not in rendered
