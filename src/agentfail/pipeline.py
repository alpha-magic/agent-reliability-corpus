"""Agent A — the weekly pipeline.

A stateless, idempotent state-machine: `scrape → classify → cross_link →
publish`. Runs under GitHub Actions on cron. Each run produces exactly one
versioned snapshot (the ISO-week revision).

Design constraints:
- No per-issue resumability yet (single-shot). Safe to re-run on failure;
  the incremental `since` filter keeps re-scrapes cheap.
- Outputs are deterministic given the same inputs + pinned model IDs.
- Everything that a paper needs to reproduce (model, prompt version,
  classifier version, revision) is recorded on each row.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from agentfail.classify import Classifier
from agentfail.crosslink import CrossLinker
from agentfail.frameworks import FRAMEWORKS
from agentfail.publish import (
    current_revision,
    push_to_hub,
    write_local_snapshot,
)
from agentfail.schema import ClassifiedIssue, CrossLink, RawIssue
from agentfail.scrape import GitHubClient, scrape_all

log = structlog.get_logger(__name__)


# --- State file: tracks last-successful scrape per framework -------------


@dataclass
class PipelineState:
    last_scraped_at: dict[str, str]  # framework slug → ISO timestamp

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        if not path.exists():
            return cls(last_scraped_at={})
        payload = json.loads(path.read_text())
        return cls(last_scraped_at=payload.get("last_scraped_at", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_scraped_at": self.last_scraped_at}, indent=2))

    def since_for(self, slug: str) -> datetime | None:
        raw = self.last_scraped_at.get(slug)
        if not raw:
            return None
        return datetime.fromisoformat(raw)

    def mark_scraped(self, slug: str, ts: datetime) -> None:
        self.last_scraped_at[slug] = ts.isoformat()


# --- Pipeline steps ------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


def run(
    *,
    output_dir: Path,
    state_path: Path,
    push: bool,
    hf_repo_id: str | None,
    max_per_framework: int | None = None,
    dry_run_classifier: bool = False,
    only_framework: str | None = None,
) -> Path:
    """Run the pipeline end-to-end; return the snapshot directory path.

    Args:
        output_dir: Where to write Parquet snapshots. Created if missing.
        state_path: JSON file holding per-framework last-scraped timestamps.
        push: If True, push the snapshot to HF Hub. Requires HF_TOKEN.
        hf_repo_id: Target HF dataset repo (e.g. "user/agent-reliability-corpus").
        max_per_framework: If set, cap issues scraped per framework. Useful
            for smoke tests; None means no cap.
        dry_run_classifier: If True, skip the LLM classifier entirely and
            emit a placeholder classification for every issue. Used by
            end-to-end tests that can't hit the DeepSeek API.
        only_framework: If set, only scrape + classify this one framework
            slug. Used by the backfill workflow to shard the pipeline
            across parallel GitHub Actions jobs (one per framework).
    """
    started_at = datetime.now(UTC)
    revision = current_revision()
    log.info("pipeline.start", revision=revision, push=push, only_framework=only_framework)

    state = PipelineState.load(state_path)

    if only_framework is not None:
        frameworks_to_run = tuple(f for f in FRAMEWORKS if f.slug == only_framework)
        if not frameworks_to_run:
            valid = sorted(f.slug for f in FRAMEWORKS)
            raise RuntimeError(f"Unknown framework slug {only_framework!r}. Valid slugs: {valid}")
    else:
        frameworks_to_run = FRAMEWORKS

    classified: list[ClassifiedIssue] = []
    classifier: Classifier | None = None
    if not dry_run_classifier:
        classifier = Classifier()

    with GitHubClient() as github:
        for framework in frameworks_to_run:
            since = state.since_for(framework.slug)
            latest_updated: datetime | None = None
            for fw_count, raw in enumerate(
                scrape_all(github, since=since, frameworks=(framework,))
            ):
                if max_per_framework and fw_count >= max_per_framework:
                    break
                latest_updated = (
                    max(latest_updated, raw.updated_at) if latest_updated else raw.updated_at
                )
                try:
                    if dry_run_classifier:
                        classified.append(_dry_run_classify(raw))
                    else:
                        assert classifier is not None
                        classified.append(classifier.classify_to_row(raw))
                except Exception as exc:  # noqa: BLE001 — log, skip row, continue
                    log.exception(
                        "pipeline.classify_error",
                        framework=framework.slug,
                        node_id=raw.node_id,
                        error=str(exc),
                    )

            if latest_updated is not None:
                state.mark_scraped(framework.slug, latest_updated)

    # Cross-linking (no-op in v0 until corpora loaders land).
    links: list[CrossLink] = CrossLinker().link_batch(classified)

    # Write snapshot + (optionally) push.
    snapshot_dir = write_local_snapshot(
        issues=classified,
        cross_links=links,
        output_dir=output_dir,
        revision=revision,
    )
    state.save(state_path)

    if push:
        if not hf_repo_id:
            raise RuntimeError("--push requires --hf-repo-id")
        push_to_hub(snapshot_dir, repo_id=hf_repo_id, revision=revision)

    duration = (datetime.now(UTC) - started_at).total_seconds()
    log.info(
        "pipeline.done",
        revision=revision,
        n_classified=len(classified),
        n_links=len(links),
        duration_s=duration,
    )
    return snapshot_dir


def _dry_run_classify(issue: RawIssue) -> ClassifiedIssue:
    """Emit a deterministic placeholder classification (for smoke/e2e tests)."""
    from agentfail import __version__
    from agentfail.schema import Classification

    placeholder = Classification(
        locus="unknown",
        phase="unknown",
        symptom="unknown",
        root_cause="unknown",
        confidence=0.0,
        reasoning="dry-run placeholder (classifier skipped)",
        needs_review=True,
    )
    return ClassifiedIssue.from_parts(
        issue,
        placeholder,
        classifier_tier="dry_run",
        classifier_model="dry-run",
        classifier_version=__version__,
        classified_at=datetime.now(UTC),
    )


# --- CLI entrypoint -----------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Agent Reliability Corpus weekly pipeline (Agent A)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Directory to write Parquet snapshots under.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("data/pipeline_state.json"),
        help="Where to read/write the per-framework scrape state.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the snapshot to the HF Hub. Requires HF_TOKEN.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="HF dataset repo ID, e.g. 'username/agent-reliability-corpus'.",
    )
    parser.add_argument(
        "--max-per-framework",
        type=int,
        default=None,
        help="Cap issues per framework (smoke-test option).",
    )
    parser.add_argument(
        "--dry-run-classifier",
        action="store_true",
        help="Skip the LLM classifier (emit placeholder rows).",
    )
    parser.add_argument(
        "--only-framework",
        default=None,
        help=(
            "Run the pipeline for a single framework slug (e.g. 'langchain'). "
            "Used by the backfill workflow to shard work across parallel jobs."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    run(
        output_dir=args.output_dir,
        state_path=args.state_path,
        push=args.push,
        hf_repo_id=args.hf_repo_id,
        max_per_framework=args.max_per_framework,
        dry_run_classifier=args.dry_run_classifier,
        only_framework=args.only_framework,
    )


if __name__ == "__main__":
    main()
