"""Classifier tests — mock the OpenAI-compatible client so no network hits.

A single `_FakeOpenAIClient` records calls and returns a configurable
sequence of tool-call responses shaped like OpenAI chat completions. The
classifier itself is single-tier (one DeepSeek V4-pro call per issue),
so these tests focus on request shape, response parsing, and argument
plumbing rather than escalation logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentfail.classify import (
    MODEL_V4_PRO,
    Classifier,
    _render_issue_for_classification,
)
from agentfail.schema import Classification

# --- Fake OpenAI client -------------------------------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    function: _FakeFunction
    type: str = "function"


@dataclass
class _FakeMessage:
    tool_calls: list[_FakeToolCall]


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "tool_calls"


@dataclass
class _FakeUsage:
    prompt_tokens: int = 1000
    completion_tokens: int = 200
    total_tokens: int = 1200
    completion_tokens_details: Any = None
    model_extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


@dataclass
class _FakeChatCompletions:
    responses: list[Classification]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> _FakeResponse:
        if not self.responses:
            raise AssertionError("FakeChatCompletions ran out of responses")
        self.calls.append(kwargs)
        next_cls = self.responses.pop(0)
        tool_call = _FakeToolCall(
            function=_FakeFunction(
                name="classify_failure",
                arguments=json.dumps(next_cls.model_dump()),
            )
        )
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(tool_calls=[tool_call]))])


@dataclass
class _FakeChat:
    completions: _FakeChatCompletions


@dataclass
class _FakeOpenAIClient:
    chat: _FakeChat


def _classifier_with_fakes(
    responses: list[Classification],
    *,
    extra_body: dict | None = None,
    model: str = MODEL_V4_PRO,
) -> tuple[Classifier, _FakeChatCompletions]:
    """Build a Classifier whose OpenAI client is replaced with a fake."""
    c = Classifier.__new__(Classifier)  # bypass __init__ (needs API key)
    fake_completions = _FakeChatCompletions(responses=list(responses))
    c._client = _FakeOpenAIClient(chat=_FakeChat(completions=fake_completions))  # type: ignore[attr-defined]
    c._model = model  # type: ignore[attr-defined]
    c._extra_body = extra_body  # type: ignore[attr-defined]
    c._use_max_completion_tokens = False  # type: ignore[attr-defined]
    c._omit_temperature = False  # type: ignore[attr-defined]
    from agentfail.taxonomy import render_taxonomy_for_prompt

    # Mirror the system-content build in __init__ so tests see the same
    # cacheable prefix the production code sends.
    c._system_content = (  # type: ignore[attr-defined]
        render_taxonomy_for_prompt()
        + "\n\n---\n\n"
        + (
            "You are a careful, conservative classifier. When in doubt, "
            "prefer `unknown` and set `needs_review=true`. Do not invent "
            "labels outside the taxonomy."
        )
    )
    return c, fake_completions


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


# --- Single-call behavior tests -----------------------------------------


def test_classify_returns_v4_pro_tier(raw_issue):
    c, fake = _classifier_with_fakes([_make_classification(confidence=0.9)])
    result = c.classify(raw_issue)
    assert result.tier == "v4_pro"
    assert result.model_id == MODEL_V4_PRO
    assert len(fake.calls) == 1


def test_classify_propagates_needs_review(raw_issue):
    # needs_review propagates straight to the output row regardless of
    # confidence — there is no escalation tier to gate it on.
    c, fake = _classifier_with_fakes([_make_classification(confidence=0.95, needs_review=True)])
    result = c.classify(raw_issue)
    assert result.classification.needs_review is True
    assert len(fake.calls) == 1


def test_classify_accepts_low_confidence_output(raw_issue):
    # Single-tier: low-confidence outputs are still accepted (no fallback).
    # Downstream filters can use confidence + needs_review to drop them.
    c, _ = _classifier_with_fakes([_make_classification(confidence=0.2, needs_review=True)])
    result = c.classify(raw_issue)
    assert result.tier == "v4_pro"
    assert result.classification.confidence == 0.2


# --- Request-shape tests ------------------------------------------------


def test_request_passes_extra_body_when_configured(raw_issue):
    """Provider-specific request kwargs (e.g. DeepSeek's thinking-disabled
    toggle) should pass through when set on the Classifier; for providers
    that don't accept them (Mistral, OpenAI, Gemini), the kwarg should be
    omitted from the request entirely so the API doesn't reject it."""
    # With extra_body configured (the DeepSeek pipeline default).
    c, fake = _classifier_with_fakes(
        [_make_classification(confidence=0.9)],
        extra_body={"thinking": {"type": "disabled"}},
    )
    c.classify(raw_issue)
    assert fake.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}

    # Without extra_body (relabel runs against Mistral/OpenAI/Gemini).
    c2, fake2 = _classifier_with_fakes([_make_classification(confidence=0.9)])
    c2.classify(raw_issue)
    assert "extra_body" not in fake2.calls[0]


def test_request_uses_temperature_zero(raw_issue):
    c, fake = _classifier_with_fakes([_make_classification(confidence=0.9)])
    c.classify(raw_issue)
    assert fake.calls[0]["temperature"] == 0


def test_request_targets_v4_pro_with_required_tool(raw_issue):
    c, fake = _classifier_with_fakes([_make_classification(confidence=0.9)])
    c.classify(raw_issue)
    sent = fake.calls[0]
    assert sent["model"] == MODEL_V4_PRO
    assert sent["tool_choice"]["function"]["name"] == "classify_failure"
    assert sent["tools"][0]["function"]["name"] == "classify_failure"


def test_system_message_is_byte_identical_across_calls(raw_issue):
    """Cache hits depend on the system prefix never changing."""
    c, fake = _classifier_with_fakes(
        [
            _make_classification(confidence=0.9),
            _make_classification(confidence=0.85),
        ]
    )
    c.classify(raw_issue)
    c.classify(raw_issue)
    sys1 = fake.calls[0]["messages"][0]["content"]
    sys2 = fake.calls[1]["messages"][0]["content"]
    assert sys1 == sys2, "system prompt drifted between calls — would defeat prefix cache"


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
