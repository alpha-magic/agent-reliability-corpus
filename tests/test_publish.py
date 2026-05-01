"""Tests for the snapshot publisher.

The HF push path is mostly network IO and gets exercised end-to-end in
the GitHub Actions backfill workflow. These tests cover the part that
broke in production: 0-row parquets. v0 ships an empty `cross_links`
config (the cross-linker is a stub), and `Dataset.from_parquet` raises
`ValueError: Instruction "train" corresponds to no data!` on those.
The publisher now constructs `Dataset` from an arrow Table directly,
which handles empty configs without complaint.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from datasets import Dataset

from agentfail.publish import push_to_hub


def _write_empty_cross_links(snapshot_dir: Path) -> None:
    """Mirror the schema cross_links_to_df([]) emits."""
    pl.DataFrame(
        schema={
            "node_id": pl.Utf8,
            "corpus_id": pl.Utf8,
            "external_id": pl.Utf8,
            "similarity": pl.Float64,
            "method": pl.Utf8,
        }
    ).write_parquet(snapshot_dir / "cross_links.parquet")


def test_dataset_from_empty_parquet_via_arrow(tmp_path: Path) -> None:
    """Direct check that the publisher's chosen construction path
    (pq.read_table → Dataset(table)) doesn't choke on 0-row inputs."""
    _write_empty_cross_links(tmp_path)
    table = pq.read_table(str(tmp_path / "cross_links.parquet"))
    ds = Dataset(table)
    assert len(ds) == 0
    # Schema is preserved — the empty config still ships the right columns
    assert set(ds.column_names) == {
        "node_id",
        "corpus_id",
        "external_id",
        "similarity",
        "method",
    }


def test_push_to_hub_includes_empty_cross_links(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: publishing a snapshot with an empty cross_links shard
    should still call push for that config (with a 0-row Dataset).

    The previous implementation raised before reaching push for any
    empty config, leaving the dataset incomplete on the Hub.
    """
    snapshot_dir = tmp_path / "2026-W18"
    snapshot_dir.mkdir()

    # Non-empty issues + frameworks + taxonomy; empty cross_links.
    pl.DataFrame(
        [
            {
                "framework_slug": "x",
                "issue_number": 1,
                "node_id": "n1",
                "title": "t",
                "body": None,
                "url": "u",
                "labels": ["bug"],
                "state": "open",
                "is_pull_request": False,
                "created_at": None,
                "updated_at": None,
                "closed_at": None,
                "comment_count": 0,
                "locus": "framework",
                "phase": "action",
                "symptom": "crash",
                "root_cause": "api_misuse",
                "confidence": 0.9,
                "reasoning": "r",
                "needs_review": False,
                "classifier_tier": "v4_pro",
                "classifier_model": "deepseek-v4-pro",
                "classifier_version": "0.1.0",
                "classified_at": None,
            }
        ]
    ).write_parquet(snapshot_dir / "issues.parquet")
    pl.DataFrame(
        [{"slug": "x", "repo": "x/x", "display_name": "X", "homepage": None}]
    ).write_parquet(snapshot_dir / "frameworks.parquet")
    pl.DataFrame(
        [{"axis": "locus", "label": "framework", "definition": "d", "derived_from": ["mast"]}]
    ).write_parquet(snapshot_dir / "taxonomy.parquet")
    _write_empty_cross_links(snapshot_dir)

    pushed_configs: list[tuple[str, int]] = []

    def fake_push(self, *, repo_id, config_name, revision, token, private):
        pushed_configs.append((config_name, len(self)))

    monkeypatch.setattr(Dataset, "push_to_hub", fake_push)

    push_to_hub(
        snapshot_dir,
        repo_id="user/agent-reliability-corpus",
        hf_token="fake-token",
        revision="2026-W18",
    )

    pushed_by_config = dict(pushed_configs)
    assert "cross_links" in pushed_by_config, "empty cross_links must still be pushed"
    assert pushed_by_config["cross_links"] == 0
    assert pushed_by_config["issues"] == 1
