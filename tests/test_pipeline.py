"""End-to-end pipeline test.

Uses `--dry-run-classifier` (so no Anthropic API call) and respx to mock
GitHub. Verifies the full scrape → classify → write-snapshot path.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import respx

from agentfail.frameworks import FRAMEWORKS
from agentfail.pipeline import PipelineState, run


@respx.mock
def test_pipeline_dry_run_produces_snapshot(tmp_path: Path, github_issue_payload):
    # Mock every framework's endpoint with the same single-issue payload —
    # makes the assertion simple and verifies we iterate all frameworks.
    for fw in FRAMEWORKS:
        respx.route(
            method="GET",
            url__startswith=f"https://api.github.com/repos/{fw.repo}/issues",
        ).mock(
            return_value=httpx.Response(
                200,
                json=[github_issue_payload],
                headers={"Link": ""},
            )
        )

    output = tmp_path / "snapshots"
    state = tmp_path / "state.json"

    snapshot_dir = run(
        output_dir=output,
        state_path=state,
        push=False,
        hf_repo_id=None,
        dry_run_classifier=True,
    )

    # Snapshot directory has the four parquet configs
    files = {p.name for p in snapshot_dir.iterdir()}
    assert files == {
        "issues.parquet",
        "cross_links.parquet",
        "taxonomy.parquet",
        "frameworks.parquet",
    }

    # Issues parquet has one row per framework (each mocked returns 1 issue)
    df = pl.read_parquet(snapshot_dir / "issues.parquet")
    assert len(df) == len(FRAMEWORKS)
    assert df["classifier_tier"].unique().to_list() == ["human"]  # dry-run marker
    assert df["classifier_model"].unique().to_list() == ["dry-run"]

    # Taxonomy parquet is populated
    tax = pl.read_parquet(snapshot_dir / "taxonomy.parquet")
    assert len(tax) > 0
    assert set(tax["axis"].unique().to_list()) == {"locus", "phase", "symptom", "root_cause"}

    # Frameworks parquet has the full roster
    fws = pl.read_parquet(snapshot_dir / "frameworks.parquet")
    assert len(fws) == len(FRAMEWORKS)

    # State file records a last-scraped timestamp for each framework
    state_data = json.loads(state.read_text())
    assert set(state_data["last_scraped_at"].keys()) == {fw.slug for fw in FRAMEWORKS}


@respx.mock
def test_pipeline_handles_empty_responses(tmp_path: Path):
    for fw in FRAMEWORKS:
        respx.route(
            method="GET",
            url__startswith=f"https://api.github.com/repos/{fw.repo}/issues",
        ).mock(return_value=httpx.Response(200, json=[], headers={"Link": ""}))

    output = tmp_path / "snapshots"
    state = tmp_path / "state.json"

    snapshot_dir = run(
        output_dir=output,
        state_path=state,
        push=False,
        hf_repo_id=None,
        dry_run_classifier=True,
    )

    df = pl.read_parquet(snapshot_dir / "issues.parquet")
    assert len(df) == 0
    # State exists even if nothing was scraped (empty object is fine)
    assert state.exists()


def test_pipeline_state_roundtrip(tmp_path: Path):
    path = tmp_path / "s.json"
    s = PipelineState.load(path)
    assert s.last_scraped_at == {}
    from datetime import UTC, datetime

    s.mark_scraped("langchain", datetime(2026, 4, 23, tzinfo=UTC))
    s.save(path)

    s2 = PipelineState.load(path)
    assert s2.since_for("langchain") is not None
    assert s2.since_for("crewai") is None
