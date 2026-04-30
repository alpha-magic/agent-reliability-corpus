"""Unified agent-failure taxonomy.

Derived from five prior academic corpora by lifting each to a common 4-axis
schema. Each label carries provenance (`derived_from`) so the cross-link step
can match our labels back to the source taxonomies.

Prior work (citation keys used in `derived_from`):
    mast            — MAST: Multi-Agent System Failure Taxonomy (Cemri et al. 2025)
    agentfail       — AgentFail: Lifecycle of Failures in Platform-Orchestrated
                      Agentic Workflows (arXiv 2509.23735)
    agent_error     — Where LLM Agents Fail and How They Learn From Failures
                      (arXiv 2509.25370)
    faults_agentic  — Characterizing Faults in Agentic AI (arXiv 2603.06847)
    framework_bugs  — An Empirical Study of Bugs in Modern LLM Agent Frameworks
                      (arXiv 2602.21806) — API misuse / incompat / doc desync
"""

from __future__ import annotations

from agentfail.schema import TaxonomyLabel

# --- Label definitions ---------------------------------------------------
# Keep labels aligned with the Literals in schema.py. Any change here needs
# a matching change there (and a deprecation note if removing).

LOCUS: tuple[TaxonomyLabel, ...] = (
    TaxonomyLabel(
        axis="locus",
        label="model",
        definition="Failure originates in the LLM itself: hallucination, wrong tool choice, token-limit behavior, or reasoning error.",
        derived_from=("agent_error", "faults_agentic"),
    ),
    TaxonomyLabel(
        axis="locus",
        label="agent",
        definition="Failure in a single agent's scaffolding: memory loss/duplication, reflection failure, planning loop, misinterpretation of state.",
        derived_from=("agent_error", "agentfail"),
    ),
    TaxonomyLabel(
        axis="locus",
        label="framework",
        definition="Framework code is at fault: API misuse, breaking changes between versions, documentation desync, or a bug in the framework itself.",
        derived_from=("framework_bugs", "faults_agentic"),
    ),
    TaxonomyLabel(
        axis="locus",
        label="workflow",
        definition="Orchestration across multiple agents/steps fails: handoff errors, role confusion, deadlock, or coordination-protocol issues.",
        derived_from=("mast", "agentfail"),
    ),
    TaxonomyLabel(
        axis="locus",
        label="platform",
        definition="Runtime or infrastructure is at fault: timeouts, API quotas, sandbox/env issues, cost overruns, deployment config.",
        derived_from=("agentfail",),
    ),
    TaxonomyLabel(
        axis="locus",
        label="unknown",
        definition="Insufficient information to attribute locus. Flagged for review.",
    ),
)

PHASE: tuple[TaxonomyLabel, ...] = (
    TaxonomyLabel(
        axis="phase",
        label="planning",
        definition="Failure during task decomposition, goal setting, or plan generation.",
        derived_from=("agent_error",),
    ),
    TaxonomyLabel(
        axis="phase",
        label="action",
        definition="Failure during tool/function call or environment interaction.",
        derived_from=("agent_error", "mast"),
    ),
    TaxonomyLabel(
        axis="phase",
        label="reflection",
        definition="Failure during self-assessment, error recovery, or course correction.",
        derived_from=("agent_error",),
    ),
    TaxonomyLabel(
        axis="phase",
        label="memory",
        definition="Failure in reading/writing short-term or long-term state.",
        derived_from=("agent_error",),
    ),
    TaxonomyLabel(
        axis="phase",
        label="coordination",
        definition="Failure during handoff, message passing, or shared-state access in a multi-agent workflow. Only applies when >1 agent is involved.",
        derived_from=("mast",),
    ),
    TaxonomyLabel(
        axis="phase",
        label="infra",
        definition="Failure in setup, teardown, or platform-level lifecycle (not within a step).",
        derived_from=("agentfail",),
    ),
    TaxonomyLabel(
        axis="phase",
        label="unknown",
        definition="Phase cannot be determined.",
    ),
)

SYMPTOM: tuple[TaxonomyLabel, ...] = (
    TaxonomyLabel(
        axis="symptom",
        label="wrong_output",
        definition="Agent produced output but it is incorrect, irrelevant, or hallucinated.",
    ),
    TaxonomyLabel(
        axis="symptom",
        label="no_output",
        definition="Agent hung, timed out, or exited without producing its expected output.",
    ),
    TaxonomyLabel(
        axis="symptom",
        label="cost_overrun",
        definition="Agent consumed substantially more tokens/calls/dollars than expected for the task.",
    ),
    TaxonomyLabel(
        axis="symptom",
        label="crash",
        definition="Exception, traceback, or process termination during execution.",
    ),
    TaxonomyLabel(
        axis="symptom",
        label="security",
        definition="Prompt injection, data exfiltration, unsafe tool invocation, or policy violation.",
    ),
    TaxonomyLabel(
        axis="symptom",
        label="loop",
        definition="Non-terminating behavior: repeated tool calls, stuck reasoning, recursive replanning.",
    ),
    TaxonomyLabel(
        axis="symptom",
        label="unknown",
        definition="Symptom not described in the issue.",
    ),
)

ROOT_CAUSE: tuple[TaxonomyLabel, ...] = (
    TaxonomyLabel(
        axis="root_cause",
        label="api_misuse",
        definition="Framework API called with wrong arguments, wrong order, or in the wrong state. Most common framework-bug category.",
        derived_from=("framework_bugs",),
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="api_incompatibility",
        definition="Breaking change between library versions; works on version A, broken on version B.",
        derived_from=("framework_bugs",),
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="doc_desync",
        definition="Documentation disagrees with actual API behavior; user followed docs but got a different result.",
        derived_from=("framework_bugs",),
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="model_limitation",
        definition="The underlying model cannot handle this input reliably: context-length, reasoning depth, tool-use format.",
        derived_from=("agent_error",),
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="prompt_issue",
        definition="Prompt template or user instructions are ambiguous, contradictory, or insufficient.",
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="data_quality",
        definition="Input data (retrieved context, tool output, user-provided) is malformed, truncated, or out of distribution.",
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="infrastructure",
        definition="Network, rate-limit, quota, sandbox, or hardware issue.",
    ),
    TaxonomyLabel(
        axis="root_cause",
        label="unknown",
        definition="Root cause not identified in the issue.",
    ),
)


ALL_LABELS: tuple[TaxonomyLabel, ...] = LOCUS + PHASE + SYMPTOM + ROOT_CAUSE


# --- Prompt rendering -----------------------------------------------------
# The classifier sees this as a cache-able system-prompt block. Keep it
# stable across weekly runs so prompt caching works (90% cost reduction on
# the cached portion). Any edit invalidates the cache.


def render_taxonomy_for_prompt() -> str:
    """Render the full taxonomy as a prompt-ready block.

    Called once per classifier instantiation and used as the cacheable
    prefix of every prompt. Output is deterministic so the cache key is
    stable across calls (DeepSeek's automatic prefix cache requires
    byte-identical prefixes).
    """

    def _fmt(axis_name: str, labels: tuple[TaxonomyLabel, ...]) -> str:
        lines = [f"## Axis: {axis_name}"]
        for label in labels:
            lines.append(f"- **{label.label}**: {label.definition}")
        return "\n".join(lines)

    blocks = [
        "# Unified Agent-Failure Taxonomy",
        "",
        "You will classify a GitHub issue from an agent framework repository against four orthogonal axes. Each axis is a closed set; pick exactly one label per axis. If an axis cannot be determined from the issue text, return `unknown`.",
        "",
        _fmt("locus (where the failure manifests)", LOCUS),
        "",
        _fmt("phase (which agent-lifecycle stage)", PHASE),
        "",
        _fmt("symptom (observable manifestation)", SYMPTOM),
        "",
        _fmt("root_cause (underlying reason, when identifiable)", ROOT_CAUSE),
        "",
        "Output rules:",
        "- Always return one label per axis.",
        "- `confidence` is your overall confidence across all four axes (0.0-1.0).",
        "- `reasoning` is one sentence (<=500 chars) citing what in the issue drove the decision.",
        "- Set `needs_review=true` when the issue does not cleanly fit any existing label, or when confidence<0.6.",
    ]
    return "\n".join(blocks)
