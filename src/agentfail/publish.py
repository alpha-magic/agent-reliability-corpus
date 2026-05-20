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
import pyarrow.parquet as pq
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


# --- Cumulative corpus merge ---------------------------------------------


def load_existing_issues(repo_id: str, token: str) -> pl.DataFrame | None:
    """Load the current `issues` config from the Hub's `main` branch.

    Returns None if the repo, the config, or its parquet files don't exist
    yet (first-ever push). Concatenates across shards if the config spans
    multiple parquet files.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    try:
        info = api.dataset_info(repo_id)
    except Exception as exc:  # noqa: BLE001 — repo may not exist yet
        log.info("publish.no_existing_repo", repo_id=repo_id, error=str(exc))
        return None

    parquets = sorted(
        s.rfilename
        for s in (info.siblings or [])
        if s.rfilename.startswith("issues/") and s.rfilename.endswith(".parquet")
    )
    if not parquets:
        log.info("publish.no_existing_issues", repo_id=repo_id)
        return None

    dfs = [
        pl.read_parquet(hf_hub_download(repo_id, fn, repo_type="dataset", token=token))
        for fn in parquets
    ]
    return pl.concat(dfs, how="vertical_relaxed")


def merge_issues_cumulative(new_df: pl.DataFrame, existing_df: pl.DataFrame | None) -> pl.DataFrame:
    """Union new classified issues onto the existing corpus.

    The corpus is cumulative: a weekly run scrapes only issues updated
    since its cursor, so its output must be *added* to the corpus, never
    substituted for it. On a `node_id` collision (an issue re-scraped
    because it was updated) the row with the later `classified_at` wins —
    the fresher classification reflects the issue's current state.
    """
    if existing_df is None or existing_df.height == 0:
        return new_df
    combined = pl.concat([existing_df, new_df], how="vertical_relaxed")
    return combined.sort("classified_at", descending=True, nulls_last=True).unique(
        subset="node_id", keep="first", maintain_order=True
    )


# --- HF Hub push ---------------------------------------------------------


def push_to_hub(
    snapshot_dir: Path,
    *,
    repo_id: str,
    hf_token: str | None = None,
    revision: str | None = None,
    private: bool = False,
    also_push_to_main: bool = True,
    cumulative: bool = True,
) -> None:
    """Push every config under snapshot_dir to the HF Hub as a multi-config
    dataset. Each config becomes a loadable config_name on the dataset page.

    Each snapshot is pushed twice by default:
      1. To its revision branch (e.g. `2026-W18`) — preserved forever, the
         citable identity of that snapshot. Papers point at this.
      2. To `main` — the default branch the dataset page renders, what
         casual `load_dataset(repo_id, config)` calls return without a
         revision argument.

    Set `also_push_to_main=False` to skip step 2 (e.g. for ad-hoc back-fills
    that shouldn't displace the current latest).

    `cumulative` (default True): the `issues` config is unioned onto the
    corpus already on the Hub before pushing — `Dataset.push_to_hub`
    *replaces* a config's files rather than appending, so a non-cumulative
    push of an incremental weekly scrape would clobber the whole corpus
    down to that one run's output. The static configs (taxonomy,
    frameworks, cross_links) are regenerated each run and replace cleanly.

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
        if config_name == "issues" and cumulative:
            # Union this run's issues onto the corpus already on the Hub.
            # Without this, push_to_hub's replace semantics would shrink
            # the corpus to just this run's incremental scrape.
            new_df = pl.read_parquet(parquet_path)
            existing_df = load_existing_issues(repo_id, token)
            merged = merge_issues_cumulative(new_df, existing_df)
            log.info(
                "publish.issues_merged",
                new_rows=new_df.height,
                existing_rows=0 if existing_df is None else existing_df.height,
                merged_rows=merged.height,
            )
            ds = Dataset(merged.to_arrow())
        else:
            # Construct via pyarrow.Table rather than Dataset.from_parquet:
            # the latter routes through ParquetBuilder, which raises
            # `Instruction "train" corresponds to no data!` on 0-row files
            # (cross_links is intentionally empty in v0). Building from an
            # arrow Table preserves the schema and handles empty configs
            # without complaint.
            table = pq.read_table(str(parquet_path))
            ds = Dataset(table)

        # 1. Revisioned push (citable, never overwritten).
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

        # 2. Main push (so the dataset page shows latest data by default).
        if also_push_to_main:
            ds.push_to_hub(
                repo_id=repo_id,
                config_name=config_name,
                token=token,
                private=private,
            )
            log.info(
                "publish.pushed_config",
                repo_id=repo_id,
                config=config_name,
                revision="main",
                rows=len(ds),
            )
