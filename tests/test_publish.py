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

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from datasets import Dataset

from agentfail import publish
from agentfail.publish import merge_issues_cumulative, push_to_hub


def _issue_row(node_id: str, classified_at: datetime | None = None, **over: object) -> dict:
    """A full issues-schema row; override any field via kwargs."""
    row: dict = {
        "framework_slug": "x",
        "issue_number": 1,
        "node_id": node_id,
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
        "classified_at": classified_at,
    }
    row.update(over)
    return row


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

    pushes: list[dict] = []

    def fake_push(self, **kwargs):
        pushes.append(
            {"config": kwargs["config_name"], "revision": kwargs.get("revision"), "rows": len(self)}
        )

    monkeypatch.setattr(Dataset, "push_to_hub", fake_push)

    push_to_hub(
        snapshot_dir,
        repo_id="user/agent-reliability-corpus",
        hf_token="fake-token",
        revision="2026-W18",
        cumulative=False,
    )

    # Hybrid pattern: each of the 4 configs gets pushed twice — once to the
    # revision branch (paper-citable), once to main (default-branch UX).
    assert len(pushes) == 8

    rev_pushes = [p for p in pushes if p["revision"] == "2026-W18"]
    main_pushes = [p for p in pushes if p["revision"] is None]
    assert len(rev_pushes) == 4
    assert len(main_pushes) == 4

    # cross_links (empty) is included in both push sets.
    rev_configs = {p["config"]: p["rows"] for p in rev_pushes}
    main_configs = {p["config"]: p["rows"] for p in main_pushes}
    assert rev_configs["cross_links"] == 0
    assert main_configs["cross_links"] == 0
    assert rev_configs["issues"] == 1
    assert main_configs["issues"] == 1


def test_push_to_hub_skip_main_when_disabled(tmp_path: Path, monkeypatch) -> None:
    """`also_push_to_main=False` should only push to the revision branch.

    Useful for ad-hoc backfill runs that shouldn't displace the current
    'latest' on the dataset page.
    """
    snapshot_dir = tmp_path / "2026-W18"
    snapshot_dir.mkdir()
    _write_empty_cross_links(snapshot_dir)
    pl.DataFrame(
        [{"slug": "x", "repo": "x/x", "display_name": "X", "homepage": None}]
    ).write_parquet(snapshot_dir / "frameworks.parquet")
    pl.DataFrame(
        [{"axis": "locus", "label": "framework", "definition": "d", "derived_from": ["mast"]}]
    ).write_parquet(snapshot_dir / "taxonomy.parquet")
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

    pushes: list[dict] = []

    def fake_push(self, **kwargs):
        pushes.append({"revision": kwargs.get("revision")})

    monkeypatch.setattr(Dataset, "push_to_hub", fake_push)

    push_to_hub(
        snapshot_dir,
        repo_id="user/agent-reliability-corpus",
        hf_token="fake-token",
        revision="2026-W18",
        also_push_to_main=False,
        cumulative=False,
    )

    # Only revisioned pushes; nothing to main.
    assert len(pushes) == 4
    assert all(p["revision"] == "2026-W18" for p in pushes)


# --- Cumulative corpus merge --------------------------------------------


def test_merge_issues_cumulative_no_existing_returns_new() -> None:
    """First-ever push: nothing on the Hub yet, so the new run is the corpus."""
    new = pl.DataFrame([_issue_row("n1"), _issue_row("n2")])
    assert merge_issues_cumulative(new, None).height == 2
    assert merge_issues_cumulative(new, new.clear()).height == 2


def test_merge_issues_cumulative_unions_disjoint_runs() -> None:
    """Net-new issues are added to the corpus, not substituted for it."""
    existing = pl.DataFrame([_issue_row("old1"), _issue_row("old2")])
    new = pl.DataFrame([_issue_row("new1")])
    merged = merge_issues_cumulative(new, existing)
    assert merged.height == 3
    assert set(merged["node_id"]) == {"old1", "old2", "new1"}


def test_merge_issues_cumulative_newer_classification_wins() -> None:
    """A re-scraped issue (same node_id) keeps the fresher classification."""
    old_ts = datetime(2026, 4, 30, tzinfo=UTC)
    new_ts = datetime(2026, 5, 17, tzinfo=UTC)
    existing = pl.DataFrame([_issue_row("n1", classified_at=old_ts, symptom="crash")])
    new = pl.DataFrame([_issue_row("n1", classified_at=new_ts, symptom="loop")])
    merged = merge_issues_cumulative(new, existing)
    assert merged.height == 1
    assert merged.row(0, named=True)["symptom"] == "loop"


def test_push_to_hub_cumulative_merges_before_push(tmp_path: Path, monkeypatch) -> None:
    """The issues config pushed to the Hub is the union with the existing
    corpus — a small incremental run must never shrink the dataset."""
    snapshot_dir = tmp_path / "2026-W21"
    snapshot_dir.mkdir()
    _write_empty_cross_links(snapshot_dir)
    pl.DataFrame(
        [{"slug": "x", "repo": "x/x", "display_name": "X", "homepage": None}]
    ).write_parquet(snapshot_dir / "frameworks.parquet")
    pl.DataFrame(
        [{"axis": "locus", "label": "framework", "definition": "d", "derived_from": ["mast"]}]
    ).write_parquet(snapshot_dir / "taxonomy.parquet")
    # This run classified a single new issue.
    pl.DataFrame([_issue_row("new1")]).write_parquet(snapshot_dir / "issues.parquet")

    # Pretend the Hub already holds a 100-issue corpus.
    existing = pl.DataFrame([_issue_row(f"old{i}") for i in range(100)])
    monkeypatch.setattr(publish, "load_existing_issues", lambda repo_id, token: existing)

    issues_push_rows: list[int] = []

    def fake_push(self, **kwargs):
        if kwargs["config_name"] == "issues":
            issues_push_rows.append(len(self))

    monkeypatch.setattr(Dataset, "push_to_hub", fake_push)

    push_to_hub(
        snapshot_dir,
        repo_id="user/agent-reliability-corpus",
        hf_token="fake-token",
        revision="2026-W21",
    )

    # 100 existing + 1 new, pushed to both revision and main.
    assert issues_push_rows == [101, 101]
