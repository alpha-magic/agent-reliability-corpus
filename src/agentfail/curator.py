"""Agent B — the curator.

A stateful, multi-step agent that handles the tasks Agent A (the pipeline)
deliberately punts on: detecting new frameworks worth scraping, auditing
classification drift, triaging dataset-repo issues, and assessing whether
accumulated changes warrant a new paper release.

Built on Pydantic AI. The agent exposes a small toolset; each tool is a
narrow, well-typed Python function the model can call. The agent decides
which tools to use based on the run's prompt; a human confirms any action
that would leave a footprint (opening a PR, publishing a new snapshot,
etc.).

Why Pydantic AI here (not raw SDK like the classifier):
    The curator runs a genuine multi-step agent loop — decide which tools
    to call, gather data, reason, maybe call more tools. That is exactly
    what Pydantic AI's Agent class is built for. In the pipeline (Agent A)
    we wanted maximum control over prompt caching and batching; here we
    want clean agent semantics.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from agentfail.classify import DEEPSEEK_BASE_URL, MODEL_V4_PRO
from agentfail.frameworks import FRAMEWORK_BY_SLUG

log = structlog.get_logger(__name__)

CURATOR_MODEL_ID = MODEL_V4_PRO  # same single model as the classifier


# --- Agent dependency bundle --------------------------------------------


@dataclass
class CuratorDeps:
    """Shared state + resources injected into every tool call.

    Keeping this explicit (not module-global) makes the curator trivially
    testable: instantiate with a different snapshot_dir pointing at test
    fixtures and run.
    """

    snapshot_dir: Path  # where pipeline snapshots live
    dataset_repo_id: str | None = None  # e.g. "user/agent-reliability-corpus"; None in tests


# --- Structured outputs -------------------------------------------------


class NewFrameworkCandidate(BaseModel):
    """One framework the curator proposes adding to the scrape list."""

    slug: str = Field(..., description="Short ID (lowercase, underscores).")
    repo: str = Field(..., description="GitHub owner/repo.")
    display_name: str
    rationale: str = Field(..., description="Why this framework warrants inclusion.")
    star_count: int | None = None


class DriftReport(BaseModel):
    """Result of a classification-drift audit."""

    sample_size: int
    agreement: float = Field(..., ge=0.0, le=1.0)
    disagreements: list[str] = Field(
        default_factory=list,
        description="node_ids where the audit classifier disagrees with the stored label.",
    )
    recommendation: Literal["no_action", "investigate", "reclassify_cohort"]
    notes: str


class PaperWorthiness(BaseModel):
    """Assessment of whether accumulated changes warrant a new paper release."""

    issues_added_since: int
    frameworks_added: int
    taxonomy_changes: int
    recommend_v2_paper: bool
    reasoning: str


class CuratorDecision(BaseModel):
    """The final decision the curator emits after its run."""

    summary: str
    new_framework_candidates: list[NewFrameworkCandidate] = Field(default_factory=list)
    drift_report: DriftReport | None = None
    paper_worthiness: PaperWorthiness | None = None
    actions_requiring_human_approval: list[str] = Field(default_factory=list)


# --- The agent ----------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Agent Reliability Corpus curator. Your job is to maintain the
quality and evolution of the dataset without shipping any change that a
human hasn't approved.

You have tools to:
- propose new frameworks to scrape (never add them yourself — only propose)
- audit classification drift on a sample of the latest snapshot
- summarize dataset growth since the last paper snapshot

Principles:
- You do NOT publish, push, or open PRs directly. You draft; humans approve.
- Prefer concrete, falsifiable claims over vague observations.
- Every proposed action goes into `actions_requiring_human_approval`.
- When a tool returns data, reason from that data — do not invent numbers.
"""


def build_curator_agent() -> Agent[CuratorDeps, CuratorDecision]:
    """Construct the Pydantic AI curator agent.

    The agent is built lazily so tests that don't exercise the LLM (and
    therefore don't need an API key) can import this module freely.

    Uses DeepSeek V4-pro via the OpenAI-compatible API. Reasoning ("thinking")
    mode is disabled to keep output token counts bounded — the curator's
    structured-output schema doesn't benefit from inline reasoning traces,
    and DeepSeek bills reasoning tokens at output rates.
    """
    provider = OpenAIProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    )
    model = OpenAIModel(
        CURATOR_MODEL_ID,
        provider=provider,
        settings=ModelSettings(extra_body={"thinking": {"type": "disabled"}}),
    )
    agent: Agent[CuratorDeps, CuratorDecision] = Agent(
        model=model,
        deps_type=CuratorDeps,
        output_type=CuratorDecision,
        system_prompt=_SYSTEM_PROMPT,
    )

    # --- Tool: propose new frameworks ------------------------------------

    @agent.tool
    def propose_new_framework(
        ctx: RunContext[CuratorDeps],
        slug: str,
        repo: str,
        display_name: str,
        rationale: str,
    ) -> str:
        """Propose adding a new agent framework to the scrape list.

        This does NOT modify the scrape list. It records the proposal so a
        human can review and commit the change to `frameworks.py`.
        """
        if slug in FRAMEWORK_BY_SLUG:
            return f"Framework '{slug}' already in the scrape list."
        log.info(
            "curator.framework_proposed",
            slug=slug,
            repo=repo,
        )
        return f"Proposal recorded: {slug} ({repo}). Requires human review."

    # --- Tool: audit classification drift --------------------------------

    @agent.tool
    def audit_classification_drift(
        ctx: RunContext[CuratorDeps],
        sample_size: int = 50,
    ) -> dict[str, object]:
        """Return summary stats for a sample of the latest classified snapshot.

        A real drift audit would re-run an audit classifier on the sampled
        rows and compare labels. This skeleton returns the per-axis label
        distribution of the sample so the curator can reason about it; the
        re-run can slot in as an extension.
        """
        snapshot_dir = ctx.deps.snapshot_dir
        # Find the latest revision under snapshot_dir/
        revisions = sorted(
            [p for p in snapshot_dir.iterdir() if p.is_dir()],
            reverse=True,
        )
        if not revisions:
            return {"error": "no snapshots under snapshot_dir"}

        latest = revisions[0]
        issues_file = latest / "issues.parquet"
        if not issues_file.exists():
            return {"error": f"no issues.parquet in {latest.name}"}

        df = pl.read_parquet(issues_file)
        if len(df) == 0:
            return {
                "revision": latest.name,
                "sample_size": 0,
                "note": "empty snapshot — nothing to audit",
            }

        sample = df.sample(n=min(sample_size, len(df)), seed=42)

        # Polars' .mean() returns a broad union type because the column dtype
        # isn't known at compile time. We know confidence is Float64 and
        # needs_review is Bool, both of which coerce cleanly to float.
        def _mean_as_float(col: str) -> float:
            v = sample[col].mean()
            return float(v) if isinstance(v, (int, float)) else 0.0

        return {
            "revision": latest.name,
            "sample_size": len(sample),
            "locus_dist": sample["locus"].value_counts().to_dicts(),
            "symptom_dist": sample["symptom"].value_counts().to_dicts(),
            "mean_confidence": _mean_as_float("confidence"),
            "needs_review_rate": _mean_as_float("needs_review"),
        }

    # --- Tool: summarize growth since a date -----------------------------

    @agent.tool
    def dataset_growth_summary(
        ctx: RunContext[CuratorDeps],
        since_iso: str | None = None,
    ) -> dict[str, object]:
        """Summarize how the dataset has grown since a date (default: 90 days ago).

        Used to assess paper-worthiness.
        """
        snapshot_dir = ctx.deps.snapshot_dir
        since = (
            datetime.fromisoformat(since_iso)
            if since_iso
            else datetime.now(UTC) - timedelta(days=90)
        )
        total = 0
        recent = 0
        for rev_dir in snapshot_dir.iterdir():
            if not rev_dir.is_dir():
                continue
            f = rev_dir / "issues.parquet"
            if not f.exists():
                continue
            df = pl.read_parquet(f)
            total += len(df)
            if len(df):
                recent += int(df.filter(pl.col("classified_at") >= since).height)
        return {
            "since": since.isoformat(),
            "total_rows_across_revisions": total,
            "classified_since": recent,
        }

    return agent


# --- Sync runner --------------------------------------------------------


async def run_curator(
    prompt: str,
    *,
    snapshot_dir: Path,
    dataset_repo_id: str | None = None,
) -> CuratorDecision:
    if "DEEPSEEK_API_KEY" not in os.environ:
        raise RuntimeError("DEEPSEEK_API_KEY required to run the curator.")

    deps = CuratorDeps(snapshot_dir=snapshot_dir, dataset_repo_id=dataset_repo_id)
    agent = build_curator_agent()
    result = await agent.run(prompt, deps=deps)
    return result.output


# --- CLI ----------------------------------------------------------------


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description="Run Agent B (the curator).")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Directory containing pipeline snapshots.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="HF dataset repo ID (for context).",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Do a routine check. Audit the latest snapshot for classification "
            "drift, summarize growth over the last 90 days, and report any "
            "new-framework candidates worth considering."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(message)s",
    )

    decision = asyncio.run(
        run_curator(
            args.prompt,
            snapshot_dir=args.snapshot_dir,
            dataset_repo_id=args.dataset_repo_id,
        )
    )
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
