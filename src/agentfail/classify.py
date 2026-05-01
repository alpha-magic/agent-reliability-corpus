"""Single-tier agent-failure classifier using DeepSeek V4-pro.

Sends each `RawIssue` to deepseek-v4-pro via the OpenAI-compatible API
and returns a structured `Classification` via tool calling. Reasoning
("thinking") mode is explicitly disabled — for a structured tool-call
classification task, reasoning tokens are pure cost overhead (DeepSeek
bills them at output rates) and provide no benefit since the output is
a small, schema-constrained JSON object.

Reproducibility choices:
- Pinned model ID (deepseek-v4-pro) recorded on every row
- temperature=0 for determinism
- thinking.type=disabled to suppress reasoning tokens
- Tool-use with strict `Classification` schema
- All four versioned fields recorded on the output row

Cost control via DeepSeek's automatic prefix caching:
- The taxonomy block (~3.6KB, byte-identical across calls) sits at the
  start of every prompt, so DeepSeek's automatic prefix cache detects
  the shared prefix and bills the cached portion at ~50× discount on
  cache hits. There is no `cache_control` marker to set; caching is
  implicit.
- Only the issue text varies per call.
- Reasoning is disabled, so output tokens are bounded to the tool-call
  payload (~200 tokens per call).

Why the OpenAI SDK, not a DeepSeek-specific client:
- DeepSeek exposes an OpenAI-compatible Chat Completions API at
  https://api.deepseek.com. Using the `openai` SDK with a custom
  base_url keeps the classifier provider-agnostic — swapping in any
  OpenAI-compatible backend (Mistral, Together, OpenRouter, a local
  vLLM server, etc.) is a one-line base_url change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)

from agentfail import __version__
from agentfail.schema import Classification, ClassifiedIssue, ClassifierTier, RawIssue
from agentfail.taxonomy import render_taxonomy_for_prompt

log = structlog.get_logger(__name__)

# --- Endpoint + pinned model --------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_V4_PRO = "deepseek-v4-pro"

# --- Tool definition ----------------------------------------------------

_TOOL_NAME = "classify_failure"

# Derive the tool's parameter schema from the Pydantic model so any schema
# change flows through automatically. `additionalProperties=False` keeps
# the classifier from inventing fields outside the closed-Literal axes.
_CLASSIFICATION_SCHEMA: dict[str, Any] = Classification.model_json_schema()
_CLASSIFICATION_SCHEMA["additionalProperties"] = False

_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Return your classification of the GitHub issue against the "
            "unified agent-failure taxonomy."
        ),
        "parameters": _CLASSIFICATION_SCHEMA,
    },
}

_TOOL_CHOICE: dict[str, Any] = {
    "type": "function",
    "function": {"name": _TOOL_NAME},
}

# Static directive appended after the (cacheable) taxonomy block.
_DIRECTIVE = (
    "You are a careful, conservative classifier. When in doubt, "
    "prefer `unknown` and set `needs_review=true`. Do not invent "
    "labels outside the taxonomy."
)


@dataclass(frozen=True)
class ClassifierResult:
    classification: Classification
    tier: ClassifierTier
    model_id: str


class Classifier:
    """Single-tier classifier hitting DeepSeek V4-pro through the OpenAI SDK.

    Constructed once per pipeline run; caches the OpenAI client and the
    rendered taxonomy block so the system message stays byte-identical
    across calls (which is what DeepSeek's automatic prefix cache keys
    off of).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        model: str = MODEL_V4_PRO,
        base_url: str = DEEPSEEK_BASE_URL,
        extra_body: dict[str, Any] | None = None,
        use_max_completion_tokens: bool = False,
        omit_temperature: bool = False,
    ) -> None:
        """Construct an OpenAI-SDK-compatible classifier.

        Args:
            api_key: explicit API key; if None, falls back to the env var
                named by `api_key_env`.
            api_key_env: env var name to read when `api_key` is None.
                Default is `DEEPSEEK_API_KEY` to match the primary
                pipeline. Set to `MISTRAL_API_KEY`, `OPENAI_API_KEY`,
                etc. when targeting a different provider for relabel /
                kappa runs.
            model: pinned model ID, recorded on every output row.
            base_url: OpenAI-compatible endpoint. DeepSeek default;
                point at any other compatible provider for cross-model
                evaluation.
            extra_body: provider-specific request kwargs passed through
                the OpenAI SDK's `extra_body`. DeepSeek V4-pro needs
                `{"thinking": {"type": "disabled"}}` to suppress
                reasoning tokens; Mistral / OpenAI / Gemini don't have
                that toggle, so leave None for them.
        """
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"{api_key_env} is required. Set it in the environment or "
                "pass api_key=... to Classifier()."
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model
        self._extra_body = extra_body
        # OpenAI's GPT-5/o-series APIs renamed `max_tokens` to
        # `max_completion_tokens` and reject the old name. DeepSeek and
        # Mistral still use `max_tokens`. Set this True when targeting
        # GPT-5.x or newer OpenAI reasoning-class models.
        self._use_max_completion_tokens = use_max_completion_tokens
        # Reasoning-class OpenAI models (GPT-5+, o-series) also reject
        # `temperature` (any value other than the default 1.0) — they
        # handle exploration internally. Set True for those.
        self._omit_temperature = omit_temperature
        # Build the system message once. Taxonomy block first (the cacheable
        # prefix), directive last. Reordering kills cache hits silently.
        self._system_content = render_taxonomy_for_prompt() + "\n\n---\n\n" + _DIRECTIVE

    # --- Single classification call -------------------------------------

    def _call(self, issue: RawIssue) -> Classification:
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_content},
                {"role": "user", "content": _render_issue_for_classification(issue)},
            ],
            "tools": [cast(ChatCompletionToolUnionParam, _TOOL)],
            "tool_choice": cast(ChatCompletionToolChoiceOptionParam, _TOOL_CHOICE),
        }
        if not self._omit_temperature:
            request_kwargs["temperature"] = 0
        if self._use_max_completion_tokens:
            request_kwargs["max_completion_tokens"] = 1024
        else:
            request_kwargs["max_tokens"] = 1024
        if self._extra_body is not None:
            # Provider-specific kwargs (e.g. DeepSeek's thinking toggle).
            # Mistral / OpenAI / Gemini reject unknown keys, so we only
            # pass extra_body when explicitly configured.
            request_kwargs["extra_body"] = self._extra_body
        response = self._client.chat.completions.create(**request_kwargs)

        # Log token usage so cost is visible per call. DeepSeek's response.usage
        # follows the OpenAI shape; provider-specific fields (cached tokens,
        # reasoning tokens) live under model_extra and are surfaced if present.
        usage = response.usage
        if usage is not None:
            extra = getattr(usage, "model_extra", None) or {}
            log.info(
                "classify.usage",
                node_id=issue.node_id,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                # DeepSeek-specific: prompt_cache_hit_tokens, prompt_cache_miss_tokens
                cache_hit_tokens=extra.get("prompt_cache_hit_tokens"),
                cache_miss_tokens=extra.get("prompt_cache_miss_tokens"),
                # Reasoning-token counters (some providers expose these)
                completion_tokens_details=getattr(usage, "completion_tokens_details", None),
            )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        for call in tool_calls:
            # openai 2.x distinguishes function tool calls from "custom" tool
            # calls. Different providers populate the discriminator field
            # differently — OpenAI sets `type="function"`, Mistral leaves
            # `type=None` while still returning a valid function tool call.
            # The reliable test is whether the `.function` attribute is
            # present and populated.
            fn = getattr(call, "function", None)
            if fn is None:
                continue
            if fn.name == _TOOL_NAME:
                return Classification.model_validate(json.loads(fn.arguments))

        raise RuntimeError(
            f"Classifier response from {self._model} did not contain a "
            f"{_TOOL_NAME} tool call. finish_reason={choice.finish_reason}, "
            f"message={choice.message!r}"
        )

    # --- Public surface --------------------------------------------------

    def classify(self, issue: RawIssue) -> ClassifierResult:
        out = self._call(issue)
        log.debug(
            "classify.complete",
            model=self._model,
            node_id=issue.node_id,
            confidence=out.confidence,
            needs_review=out.needs_review,
        )
        return ClassifierResult(out, "v4_pro", self._model)

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


# --- Helpers ------------------------------------------------------------

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
