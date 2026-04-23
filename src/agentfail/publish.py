"""Publishes the weekly corpus snapshot.

Writes local Parquet files for every dataset config, then (optionally)
pushes to the Hugging Face Hub as a multi-config dataset. Each weekly
run is committed to the Hub as its own revision — `datasets` uses git
under the hood, so every snapshot is reproducible by revision hash.

Configs published:
    issues        — one row per classified GitHub issue (primary table)
    cross_links   — links from issues to academic corpora (empty in v0)
    taxonomy      — reference labels + definitions + citations
    frameworks    — framework metadata (what we scraped)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import structlog
from datasets import Dataset

from agentfail.frameworks import FRAMEWORKS
from agentfail.schema import (
    ClassifiedIssue,
    CrossLink,
    TaxonomyLabel,
)
from agentfail.taxonomy import ALL_LABELS

log = structlog.get_logger(__name__)


# --- DataFrame builders (polars, strict schemas) -------------------------


def issues_to_df(issues: Sequence[ClassifiedIssue]) -> pl.DataFrame:
    if not issues:
        # Return an empty DataFrame with the correct schema so downstream
        # joins and writes don't break on empty weeks.
        return pl.DataFrame(
            schema={
                "framework_slug": pl.Utf8,
                "issue_number": pl.Int64,
                "node_id": pl.Utf8,
                "title": pl.Utf8,
                "body": pl.Utf8,
                "url": pl.Utf8,
                "labels": pl.List(pl.Utf8),
                "state": pl.Utf8,
                "is_pull_request": pl.Boolean,
                "created_at": pl.Datetime("us", "UTC"),
                "updated_at": pl.Datetime("us", "UTC"),
                "closed_at": pl.Datetime("us", "UTC"),
                "comment_count": pl.Int64,
                "locus": pl.Utf8,
                "phase": pl.Utf8,
                "symptom": pl.Utf8,
                "root_cause": pl.Utf8,
                "confidence": pl.Float64,
                "reasoning": pl.Utf8,
                "needs_review": pl.Boolean,
                "classifier_tier": pl.Utf8,
                "classifier_model": pl.Utf8,
                "classifier_version": pl.Utf8,
                "classified_at": pl.Datetime("us", "UTC"),
            }
        )

    rows = [
        {
            "framework_slug": issue.framework_slug,
            "issue_number": issue.issue_number,
            "node_id": issue.node_id,
            "title": issue.title,
            "body": issue.body,
            "url": str(issue.url),
            "labels": list(issue.labels),
            "state": issue.state,
            "is_pull_request": issue.is_pull_request,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "closed_at": issue.closed_at,
            "comment_count": issue.comment_count,
            "locus": issue.locus,
            "phase": issue.phase,
            "symptom": issue.symptom,
            "root_cause": issue.root_cause,
            "confidence": issue.confidence,
            "reasoning": issue.reasoning,
            "needs_review": issue.needs_review,
            "classifier_tier": issue.classifier_tier,
            "classifier_model": issue.classifier_model,
            "classifier_version": issue.classifier_version,
            "classified_at": issue.classified_at,
        }
        for issue in issues
    ]
    return pl.DataFrame(rows)


def cross_links_to_df(links: Sequence[CrossLink]) -> pl.DataFrame:
    if not links:
        return pl.DataFrame(
            schema={
                "node_id": pl.Utf8,
                "corpus_id": pl.Utf8,
                "external_id": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    return pl.DataFrame(
        [
            {
                "node_id": lk.node_id,
                "corpus_id": lk.corpus_id,
                "external_id": lk.external_id,
                "similarity": lk.similarity,
                "method": lk.method,
            }
            for lk in links
        ]
    )


def taxonomy_to_df(labels: Sequence[TaxonomyLabel] = ALL_LABELS) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "axis": lbl.axis,
                "label": lbl.label,
                "definition": lbl.definition,
                "derived_from": list(lbl.derived_from),
            }
            for lbl in labels
        ]
    )


def frameworks_to_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "slug": f.slug,
                "repo": f.repo,
                "display_name": f.display_name,
                "homepage": str(f.homepage) if f.homepage else None,
            }
            for f in FRAMEWORKS
        ]
    )


# --- Snapshot writer -----------------------------------------------------


def current_revision() -> str:
    """ISO week tag for the snapshot, e.g. '2026-W17'."""
    now = datetime.now(UTC)
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def write_local_snapshot(
    *,
    issues: Sequence[ClassifiedIssue],
    cross_links: Sequence[CrossLink],
    output_dir: Path,
    revision: str | None = None,
) -> Path:
    """Write Parquet files for every config under output_dir/<revision>/.

    Returns the snapshot directory path. Intended as the artifact that the
    HF push step later consumes; splitting write + push makes offline runs
    and testing easy.
    """
    rev = revision or current_revision()
    snapshot_dir = output_dir / rev
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    issues_to_df(issues).write_parquet(snapshot_dir / "issues.parquet")
    cross_links_to_df(cross_links).write_parquet(snapshot_dir / "cross_links.parquet")
    taxonomy_to_df().write_parquet(snapshot_dir / "taxonomy.parquet")
    frameworks_to_df().write_parquet(snapshot_dir / "frameworks.parquet")

    log.info(
        "publish.snapshot_written",
        revision=rev,
        path=str(snapshot_dir),
        n_issues=len(issues),
        n_cross_links=len(cross_links),
    )
    return snapshot_dir


# --- HF Hub push ---------------------------------------------------------


def push_to_hub(
    snapshot_dir: Path,
    *,
    repo_id: str,
    hf_token: str | None = None,
    revision: str | None = None,
    private: bool = False,
) -> None:
    """Push every config under snapshot_dir to the HF Hub as a multi-config
    dataset. Each config becomes a loadable config_name on the dataset page.

    Requires `HF_TOKEN` (or the `hf_token` argument). No-op if the snapshot
    directory contains no parquet files.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to push to the Hub. Set it in the "
            "environment or pass hf_token=... to push_to_hub()."
        )

    rev = revision or current_revision()
    configs = {
        "issues": snapshot_dir / "issues.parquet",
        "cross_links": snapshot_dir / "cross_links.parquet",
        "taxonomy": snapshot_dir / "taxonomy.parquet",
        "frameworks": snapshot_dir / "frameworks.parquet",
    }

    for config_name, parquet_path in configs.items():
        if not parquet_path.exists():
            log.warning("publish.missing_config", config=config_name, path=str(parquet_path))
            continue
        ds = Dataset.from_parquet(str(parquet_path))
        ds.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            revision=rev,
            token=token,
            private=private,
        )
        log.info(
            "publish.pushed_config",
            repo_id=repo_id,
            config=config_name,
            revision=rev,
            rows=len(ds),
        )
