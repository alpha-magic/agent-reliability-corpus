"""Tests for the per-framework shard merger.

The merge step runs in the GitHub Actions backfill workflow after every
matrix job uploads its shard artifact. These tests fabricate the shape of
those artifacts on disk and verify the merge produces a single canonical
snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from agentfail.merge import merge_shards, merge_state_files
from agentfail.pipeline import run as run_pipeline
from agentfail.publish import current_revision


def _make_shard(
    shards_dir: Path,
    framework_slug: str,
    issues_rows: list[dict],
    state_payload: dict,
) -> Path:
    """Fabricate one shard layout matching what a matrix job uploads."""
    revision = current_revision()
    shard = shards_dir / f"shard-{framework_slug}"
    snapshot = shard / "snapshots" / revision
    snapshot.mkdir(parents=True)
    pl.DataFrame(issues_rows).write_parquet(snapshot / "issues.parquet")
    (shard / "pipeline_state.json").write_text(json.dumps(state_payload))
    return shard


def _row(framework: str, issue_number: int) -> dict:
    """Minimal shape matching the publish.issues_to_df schema."""
    from datetime import UTC, datetime

    now = datetime(2026, 4, 23, tzinfo=UTC)
    return {
        "framework_slug": framework,
        "issue_number": issue_number,
        "node_id": f"node-{framework}-{issue_number}",
        "title": "t",
        "body": "b",
        "url": "https://example.com",
        "labels": [],
        "state": "open",
        "is_pull_request": False,
        "created_at": now,
        "updated_at": now,
        "closed_at": None,
        "comment_count": 0,
        "locus": "framework",
        "phase": "action",
        "symptom": "wrong_output",
        "root_cause": "api_misuse",
        "confidence": 0.9,
        "reasoning": "r",
        "needs_review": False,
        "classifier_tier": "v4_pro",
        "classifier_model": "deepseek-v4-pro",
        "classifier_version": "0.1.0",
        "classified_at": now,
    }


def test_merge_concatenates_issues_across_shards(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_shard(
        shards_dir,
        "langchain",
        [_row("langchain", 1), _row("langchain", 2)],
        {"last_scraped_at": {"langchain": "2026-04-23T10:00:00+00:00"}},
    )
    _make_shard(
        shards_dir,
        "crewai",
        [_row("crewai", 100)],
        {"last_scraped_at": {"crewai": "2026-04-23T11:00:00+00:00"}},
    )

    out_dir = tmp_path / "merged"
    state_path = tmp_path / "merged_state.json"
    snapshot = merge_shards(
        shards_dir=shards_dir,
        output_dir=out_dir,
        state_path=state_path,
    )

    df = pl.read_parquet(snapshot / "issues.parquet")
    assert df.height == 3
    assert set(df["framework_slug"].unique().to_list()) == {"langchain", "crewai"}

    # Static configs are present and non-empty.
    for name in ("frameworks.parquet", "taxonomy.parquet", "cross_links.parquet"):
        assert (snapshot / name).exists()
    assert pl.read_parquet(snapshot / "frameworks.parquet").height > 0
    assert pl.read_parquet(snapshot / "taxonomy.parquet").height > 0

    # State unioned across shards.
    state = json.loads(state_path.read_text())["last_scraped_at"]
    assert set(state.keys()) == {"langchain", "crewai"}


def test_merge_state_files_handles_missing_or_bad_files(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps({"last_scraped_at": {"a": "2026-04-23T10:00:00+00:00"}}))
    bad.write_text("not json")
    merged = merge_state_files([good, bad])
    assert merged == {"a": "2026-04-23T10:00:00+00:00"}


def test_merge_rejects_inconsistent_revisions(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    # Hand-roll two shards with conflicting revision dirs.
    s1 = shards_dir / "shard-a" / "snapshots" / "2026-W17"
    s2 = shards_dir / "shard-b" / "snapshots" / "2026-W18"
    s1.mkdir(parents=True)
    s2.mkdir(parents=True)
    pl.DataFrame([_row("a", 1)]).write_parquet(s1 / "issues.parquet")
    pl.DataFrame([_row("b", 1)]).write_parquet(s2 / "issues.parquet")

    with pytest.raises(RuntimeError, match="Inconsistent revisions"):
        merge_shards(shards_dir=shards_dir, output_dir=tmp_path / "out")


def test_merge_raises_when_no_shards(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No issues.parquet"):
        merge_shards(shards_dir=empty, output_dir=tmp_path / "out")


# --- pipeline.run --only_framework filter ----------------------------------


def test_pipeline_only_framework_filters_to_one(tmp_path: Path, monkeypatch) -> None:
    """`--only-framework langchain` should scrape and classify just langchain."""
    import respx
    from httpx import Response

    from agentfail.frameworks import FRAMEWORKS

    issue_payload = {
        "number": 1,
        "node_id": "I_kwDOAB12_5M6F7g8",
        "title": "Something fails",
        "body": "Stack trace shows a crash.",
        "html_url": "https://github.com/example/repo/issues/1",
        "labels": [{"name": "bug"}],
        "state": "open",
        "created_at": "2026-04-01T09:00:00Z",
        "updated_at": "2026-04-02T09:00:00Z",
        "closed_at": None,
        "comments": 1,
    }

    with respx.mock(assert_all_called=False) as mock:
        # Mock every framework's GitHub endpoint with the same single issue;
        # the `--only-framework` filter should still drop 11 of them.
        for fw in FRAMEWORKS:
            mock.route(
                method="GET",
                url__startswith=f"https://api.github.com/repos/{fw.repo}/issues",
            ).mock(return_value=Response(200, json=[issue_payload]))

        out_dir = tmp_path / "snapshots"
        state_path = tmp_path / "state.json"

        run_pipeline(
            output_dir=out_dir,
            state_path=state_path,
            push=False,
            hf_repo_id=None,
            dry_run_classifier=True,
            only_framework="langchain",
        )

        revision_dirs = list(out_dir.iterdir())
        assert len(revision_dirs) == 1
        df = pl.read_parquet(revision_dirs[0] / "issues.parquet")
        assert df.height == 1
        assert df["framework_slug"].unique().to_list() == ["langchain"]


def test_pipeline_only_framework_rejects_unknown_slug(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unknown framework slug"):
        run_pipeline(
            output_dir=tmp_path / "snapshots",
            state_path=tmp_path / "state.json",
            push=False,
            hf_repo_id=None,
            dry_run_classifier=True,
            only_framework="this-framework-does-not-exist",
        )
