"""Single-tier agent-failure classifier built on Pydantic AI.

Every issue is labeled by a single LLM (default: DeepSeek V4-pro) via
the Pydantic AI `Agent` abstraction. The agent's `output_type` is the
`Classification` Pydantic model, which Pydantic AI converts to a
provider-appropriate structured-output mechanism (tool calling on
OpenAI/DeepSeek/Mistral, JSON mode where tools aren't supported).

Why Pydantic AI rather than the OpenAI SDK directly:
    Each provider exposes a slightly different "OpenAI-compatible" API
    surface — Mistral leaves `tool_calls[i].type=None` while OpenAI
    sets it to `"function"`; OpenAI's GPT-5+ renamed `max_tokens` to
    `max_completion_tokens` and rejects `temperature` values other
    than the default; Llama 4 deployments on some providers don't
    expose tool calling at all. Pydantic AI's per-provider model
    classes handle these quirks internally so the classification code
    is the same regardless of who's serving the inference.

Reproducibility choices:
- Pinned model ID recorded on every output row.
- Optional `extra_body` / temperature toggles for provider-specific
  settings (DeepSeek V4-pro thinking-disabled, etc.).
- Tool-use / structured-output enforced via the `Classification`
  Pydantic model.

Cost control via prefix caching:
- The taxonomy block (~3.6KB, byte-identical across calls) sits at
  the start of every system prompt. Every major provider's prefix
  cache picks this up automatically when calls happen within their
  respective TTL windows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from agentfail import __version__
from agentfail.schema import Classification, ClassifiedIssue, ClassifierTier, RawIssue
from agentfail.taxonomy import render_taxonomy_for_prompt

log = structlog.get_logger(__name__)

# --- Endpoint + pinned model --------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_V4_PRO = "deepseek-v4-pro"

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
    """Pydantic-AI-backed single-tier classifier.

    Constructed once per pipeline run; caches the agent and the
    rendered taxonomy block so the system message stays byte-identical
    across calls (which is what every provider's prefix cache keys
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
        omit_temperature: bool = False,
        max_tokens: int | None = None,
        provider_kind: str | None = None,
    ) -> None:
        """Construct a classifier.

        Args:
            api_key: explicit API key; if None, falls back to env var
                named by `api_key_env`.
            api_key_env: env var holding the API key. Defaults to
                `DEEPSEEK_API_KEY` for the primary pipeline. Use
                `OPENAI_API_KEY`, `MISTRAL_API_KEY`, etc. for relabel
                / kappa runs against other providers.
            model: pinned model ID. Recorded on every output row.
            base_url: OpenAI-compatible chat-completions endpoint.
                Default DeepSeek's. Set to `https://api.openai.com/v1`,
                `https://api.mistral.ai/v1`, etc. for other providers.
            extra_body: provider-specific request kwargs. DeepSeek
                V4-pro takes `{"thinking": {"type": "disabled"}}` to
                suppress reasoning tokens. Other providers reject
                unknown keys; leave None for them.
            omit_temperature: skip the `temperature` setting entirely.
                Required for GPT-5.x and o-series models, which only
                accept the default `1.0`.
            max_tokens: optional cap per response. Default None — the
                Classification schema bounds output to a small
                structured payload, so a max_tokens cap is mostly
                redundant. Pydantic AI translates `max_tokens` to
                `max_completion_tokens` on every OpenAI-compatible
                provider, but Mistral's "OpenAI-compatible" endpoint
                rejects `max_completion_tokens` as an unknown param —
                so we leave it unset by default to keep the
                cross-provider invariant.
            provider_kind: which Pydantic AI provider integration to
                use. `"openai"` (default for OpenAI / DeepSeek / any
                OpenAI-Chat-Completions-shaped endpoint) or
                `"mistral"` (uses Mistral's native API, which handles
                their tool-call response shape correctly — the
                `type=None` case OpenAI-compat clients can't parse).
                If None, inferred from base_url.
        """
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"{api_key_env} is required. Set it in the environment or "
                "pass api_key=... to Classifier()."
            )

        # Auto-detect provider kind from base_url if not specified.
        # Mistral's API needs MistralModel because its tool-call
        # response shape (`type=None`) doesn't validate against the
        # OpenAI SDK's strict pydantic schema.
        if provider_kind is None:
            provider_kind = "mistral" if "mistral.ai" in base_url else "openai"

        settings_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            settings_kwargs["max_tokens"] = max_tokens
        if not omit_temperature:
            settings_kwargs["temperature"] = 0
        if extra_body:
            settings_kwargs["extra_body"] = extra_body
        settings = ModelSettings(**settings_kwargs) if settings_kwargs else None

        # Pydantic AI's per-provider Model classes type their first arg
        # as a Literal of known model names. Our model IDs are runtime
        # strings — may be valid but pyright can't prove it statically.
        pydantic_model: OpenAIChatModel | MistralModel
        if provider_kind == "mistral":
            # MistralProvider wraps the official `mistralai` SDK, which
            # has its own base_url defaulting to `https://api.mistral.ai`
            # and appends paths internally. Passing our OpenAI-compat
            # `https://api.mistral.ai/v1` causes path duplication and a
            # 404 — so we don't forward base_url to MistralProvider.
            pydantic_model = MistralModel(
                cast(Any, model),
                provider=MistralProvider(api_key=key),
            )
        else:
            pydantic_model = cast(
                OpenAIChatModel,
                OpenAIChatModel(
                    cast(Any, model),
                    provider=OpenAIProvider(base_url=base_url, api_key=key),
                ),
            )

        # Build the system prompt once. Taxonomy block first (the
        # cacheable prefix), directive last. Reordering would defeat
        # every provider's prefix cache silently.
        system_prompt = render_taxonomy_for_prompt() + "\n\n---\n\n" + _DIRECTIVE

        self._agent: Agent[None, Classification] = Agent(
            model=pydantic_model,
            output_type=Classification,
            system_prompt=system_prompt,
            model_settings=settings,
            retries=2,
        )
        self._model = model

    # --- Single classification call -------------------------------------

    def _call(self, issue: RawIssue) -> Classification:
        result = self._agent.run_sync(_render_issue_for_classification(issue))

        # Log token usage when the provider exposes it. Pydantic AI's
        # Usage object exposes `request_tokens`, `response_tokens`,
        # `total_tokens`. Provider-specific extras (DeepSeek's
        # cache_hit_tokens, etc.) are in `usage.details`.
        usage = result.usage()
        details = usage.details or {}
        log.info(
            "classify.usage",
            node_id=issue.node_id,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cache_hit_tokens=details.get("prompt_cache_hit_tokens"),
            cache_miss_tokens=details.get("prompt_cache_miss_tokens"),
        )
        return result.output

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
