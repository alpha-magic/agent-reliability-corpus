"""Classifier tests — mock the Anthropic client so no network hits.

Focused on the tier-escalation logic; the underlying Anthropic call is the
narrow seam we mock. A single `FakeAnthropicClient` records calls and
returns a configurable sequence of tool_use responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentfail.classify import (
    HAIKU_ACCEPT_THRESHOLD,
    SONNET_ACCEPT_THRESHOLD,
    Classifier,
    _render_issue_for_classification,
)
from agentfail.schema import Classification

# --- Fake Anthropic client ----------------------------------------------


@dataclass
class _FakeToolUseBlock:
    type: str
    name: str
    input: dict[str, Any]


@dataclass
class _FakeMessage:
    content: list[_FakeToolUseBlock]


@dataclass
class _FakeMessages:
    responses: list[Classification]
    calls: list[str] = field(default_factory=list)  # records model IDs called

    def create(self, **kwargs: Any) -> _FakeMessage:
        if not self.responses:
            raise AssertionError("FakeMessages ran out of responses")
        self.calls.append(kwargs["model"])
        next_cls = self.responses.pop(0)
        block = _FakeToolUseBlock(
            type="tool_use",
            name="classify_failure",
            input=next_cls.model_dump(),
        )
        return _FakeMessage(content=[block])


@dataclass
class _FakeAnthropicClient:
    messages: _FakeMessages


def _classifier_with_fakes(responses: list[Classification]) -> tuple[Classifier, _FakeMessages]:
    """Build a Classifier whose Anthropic client is replaced with a fake."""
    c = Classifier.__new__(Classifier)  # bypass __init__ (needs API key)
    fake_messages = _FakeMessages(responses=list(responses))
    c._client = _FakeAnthropicClient(messages=fake_messages)  # type: ignore[attr-defined]
    # Still need the taxonomy prompt + models dict
    from agentfail.classify import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET
    from agentfail.taxonomy import render_taxonomy_for_prompt

    c._taxonomy_prompt = render_taxonomy_for_prompt()  # type: ignore[attr-defined]
    c._models = {  # type: ignore[attr-defined]
        "haiku": MODEL_HAIKU,
        "sonnet": MODEL_SONNET,
        "opus": MODEL_OPUS,
    }
    return c, fake_messages


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


# --- Escalation tests ---------------------------------------------------


def test_haiku_confident_result_is_accepted(raw_issue):
    c, fake = _classifier_with_fakes(
        [_make_classification(confidence=HAIKU_ACCEPT_THRESHOLD + 0.05)]
    )
    result = c.classify(raw_issue)
    assert result.tier == "haiku"
    assert len(fake.calls) == 1


def test_low_haiku_confidence_escalates_to_sonnet(raw_issue):
    c, fake = _classifier_with_fakes(
        [
            _make_classification(confidence=HAIKU_ACCEPT_THRESHOLD - 0.1),
            _make_classification(confidence=SONNET_ACCEPT_THRESHOLD + 0.05),
        ]
    )
    result = c.classify(raw_issue)
    assert result.tier == "sonnet"
    assert len(fake.calls) == 2
    # The escalation called a different model
    assert fake.calls[0] != fake.calls[1]


def test_needs_review_escalates_even_with_high_confidence(raw_issue):
    # Haiku returns high confidence but flags needs_review — still escalate.
    c, fake = _classifier_with_fakes(
        [
            _make_classification(confidence=0.95, needs_review=True),
            _make_classification(confidence=SONNET_ACCEPT_THRESHOLD + 0.05),
        ]
    )
    result = c.classify(raw_issue)
    assert result.tier == "sonnet"
    assert len(fake.calls) == 2


def test_low_sonnet_confidence_escalates_to_opus(raw_issue):
    c, fake = _classifier_with_fakes(
        [
            _make_classification(confidence=HAIKU_ACCEPT_THRESHOLD - 0.2),
            _make_classification(confidence=SONNET_ACCEPT_THRESHOLD - 0.1),
            _make_classification(
                confidence=0.4, needs_review=True
            ),  # opus: still low, but accepted
        ]
    )
    result = c.classify(raw_issue)
    assert result.tier == "opus"
    assert len(fake.calls) == 3
    # Opus output is still accepted even when low-confidence; needs_review will propagate.
    assert result.classification.needs_review is True


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
