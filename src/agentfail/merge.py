"""Merge per-framework shards produced by the parallel backfill workflow.

Each parallel job in `.github/workflows/backfill.yml` runs the pipeline for
exactly one framework (`arc-pipeline --only-framework <slug>`) and uploads
its snapshot directory + state file as an artifact. After all jobs finish,
the merge job downloads every artifact into a `shards-dir/` and runs
`arc-merge-shards` to:

  1. Concatenate every shard's `issues.parquet` into a single primary table.
  2. Regenerate the static configs (`taxonomy`, `frameworks`, empty
     `cross_links`) directly from source — they're identical across shards
     so we don't trust the per-shard copies.
  3. Union every shard's `last_scraped_at` map into a single `state.json`
     so subsequent weekly cron runs only fetch deltas.
  4. Optionally push the merged snapshot to the Hugging Face Hub.

The merge step is intentionally separate from the classifier so a partial
failure (one framework's job crashes) doesn't void the rest of the run —
re-running just the failing matrix entry is cheap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl
import structlog

from agentfail.publish import (
    cross_links_to_df,
    frameworks_to_df,
    push_to_hub,
    taxonomy_to_df,
)

log = structlog.get_logger(__name__)


def _find_issues_parquet_files(shards_dir: Path) -> list[Path]:
    """All `issues.parquet` files anywhere under shards_dir, sorted for
    deterministic concat order."""
    return sorted(shards_dir.rglob("issues.parquet"))


def _find_state_files(shards_dir: Path) -> list[Path]:
    """Per-shard pipeline_state.json files (one per matrix job)."""
    return sorted(shards_dir.rglob("pipeline_state.json"))


def merge_state_files(state_files: list[Path]) -> dict[str, str]:
    """Union the `last_scraped_at` maps from every shard.

    Each shard touches only its own framework's key, so a plain dict union
    is the right merge operation. If two shards somehow disagree on the
    same key (shouldn't happen), the lexically-later file wins — sorted
    iteration makes that deterministic.
    """
    merged: dict[str, str] = {}
    for path in state_files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("merge.skip_bad_state", path=str(path), error=str(exc))
            continue
        merged.update(payload.get("last_scraped_at", {}))
    return merged


def merge_shards(
    *,
    shards_dir: Path,
    output_dir: Path,
    state_path: Path | None = None,
) -> Path:
    """Merge per-framework shards into a single canonical snapshot.

    Returns the merged snapshot directory path. Static configs (taxonomy,
    frameworks) are regenerated from source; per-shard copies are ignored.
    """
    issues_files = _find_issues_parquet_files(shards_dir)
    if not issues_files:
        raise RuntimeError(f"No issues.parquet found under {shards_dir}")

    # The revision is the parent directory name of any issues.parquet
    # (e.g. .../shards/shard-langchain/2026-W18/issues.parquet → "2026-W18").
    revisions = {p.parent.name for p in issues_files}
    if len(revisions) != 1:
        raise RuntimeError(
            f"Inconsistent revisions across shards: {sorted(revisions)}. "
            "All matrix jobs should run within the same ISO week."
        )
    revision = revisions.pop()

    snapshot_dir = output_dir / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Concatenate per-framework issues parquets.
    issues_dfs = [pl.read_parquet(f) for f in issues_files]
    merged_issues = pl.concat(issues_dfs, how="vertical_relaxed")
    merged_issues.write_parquet(snapshot_dir / "issues.parquet")

    # 2. Regenerate static configs directly from source. Per-shard copies
    #    are byte-identical (frameworks.py and taxonomy.py are static), so
    #    skipping them avoids any drift if a shard somehow shipped a stale
    #    one.
    frameworks_to_df().write_parquet(snapshot_dir / "frameworks.parquet")
    taxonomy_to_df().write_parquet(snapshot_dir / "taxonomy.parquet")
    cross_links_to_df([]).write_parquet(snapshot_dir / "cross_links.parquet")

    # 3. Optionally union per-shard state files.
    if state_path is not None:
        merged_state = merge_state_files(_find_state_files(shards_dir))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"last_scraped_at": merged_state}, indent=2, sort_keys=True)
        )
        log.info(
            "merge.state_written",
            path=str(state_path),
            n_frameworks=len(merged_state),
        )

    log.info(
        "merge.snapshot_written",
        revision=revision,
        path=str(snapshot_dir),
        n_issues=merged_issues.height,
        n_shards=len(issues_files),
    )
    return snapshot_dir


# --- CLI -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-framework shards from the backfill workflow into a "
            "single canonical snapshot, optionally pushing to the HF Hub."
        )
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        required=True,
        help="Directory containing the downloaded artifacts (one subdir per shard).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Where to write the merged snapshot (under <output-dir>/<revision>/).",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help=(
            "If set, union every shard's pipeline_state.json and write the "
            "result here. Lets weekly cron pick up incrementally after a "
            "backfill."
        ),
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the merged snapshot to the HF Hub. Requires HF_TOKEN.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="HF dataset repo ID, e.g. 'username/agent-reliability-corpus'.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(message)s",
    )

    snapshot_dir = merge_shards(
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        state_path=args.state_path,
    )

    if args.push:
        if not args.hf_repo_id:
            raise RuntimeError("--push requires --hf-repo-id")
        push_to_hub(snapshot_dir, repo_id=args.hf_repo_id, revision=snapshot_dir.name)


if __name__ == "__main__":
    main()
