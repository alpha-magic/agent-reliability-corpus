"""Tests for the relabel + κ evaluation tooling.

The full relabel CLI is not unit-tested (it routes through the real
OpenAI SDK and is exercised via integration runs). We focus on the
two pieces with all the actual logic: stratified sampling and the
κ-report computation.
"""

from __future__ import annotations

import polars as pl
import pytest

from agentfail.eval import _row_to_raw_issue, kappa_report, stratified_sample

# --- Stratified sampling ------------------------------------------------


def test_stratified_sample_returns_full_df_when_n_exceeds_size() -> None:
    df = pl.DataFrame({"framework_slug": ["a", "b"], "x": [1, 2]})
    out = stratified_sample(df, n=10, stratify_by="framework_slug")
    assert out.height == 2


def test_stratified_sample_preserves_every_group() -> None:
    """Even tiny groups get at least one row when n >= n_groups."""
    rows = [{"framework_slug": "big", "x": i} for i in range(900)] + [
        {"framework_slug": "tiny", "x": i} for i in range(2)
    ]
    df = pl.DataFrame(rows)
    out = stratified_sample(df, n=20, stratify_by="framework_slug", seed=1)
    groups = set(out["framework_slug"].unique().to_list())
    assert "tiny" in groups
    assert "big" in groups
    assert out.height == 20


def test_stratified_sample_proportional_allocation() -> None:
    """Big groups take more rows than small groups, roughly proportionally."""
    rows = (
        [{"framework_slug": "a", "x": i} for i in range(500)]
        + [{"framework_slug": "b", "x": i} for i in range(500)]
        + [{"framework_slug": "c", "x": i} for i in range(50)]
    )
    df = pl.DataFrame(rows)
    out = stratified_sample(df, n=120, stratify_by="framework_slug", seed=42)
    counts = dict(out.group_by("framework_slug").len().rows())
    assert out.height == 120
    # a and b should each get roughly 55, c roughly 6 (with floor-of-1 logic).
    assert 40 <= counts["a"] <= 70
    assert 40 <= counts["b"] <= 70
    assert counts["c"] >= 1


def test_stratified_sample_no_stratify_falls_back_to_random() -> None:
    df = pl.DataFrame({"framework_slug": ["a"] * 100, "x": list(range(100))})
    out = stratified_sample(df, n=10, stratify_by=None, seed=42)
    assert out.height == 10


# --- κ report -----------------------------------------------------------


def _label_df(node_ids: list[str], labels: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame({"node_id": node_ids, **labels})


def test_kappa_perfect_agreement() -> None:
    """Identical labels on both sides → κ = 1.0 per axis."""
    nodes = [f"n{i}" for i in range(20)]
    labels = {
        "locus": ["framework"] * 10 + ["model"] * 10,
        "phase": ["action"] * 5 + ["planning"] * 15,
        "symptom": ["crash"] * 8 + ["wrong_output"] * 12,
        "root_cause": ["api_misuse"] * 14 + ["unknown"] * 6,
    }
    primary = _label_df(nodes, labels)
    secondary = _label_df(nodes, labels)
    report = kappa_report(primary=primary, secondary=secondary)
    assert report["n"] == 20
    for axis in ("locus", "phase", "symptom", "root_cause"):
        assert report["per_axis"][axis]["kappa"] == pytest.approx(1.0)
        assert report["per_axis"][axis]["agreement_count"] == 20


def test_kappa_systematic_disagreement_yields_negative_score() -> None:
    """When the second annotator flips every label, κ should drop sharply."""
    nodes = [f"n{i}" for i in range(20)]
    primary = _label_df(
        nodes,
        {
            "locus": ["framework"] * 10 + ["model"] * 10,
            "phase": ["action"] * 20,
            "symptom": ["crash"] * 20,
            "root_cause": ["api_misuse"] * 20,
        },
    )
    secondary = _label_df(
        nodes,
        {
            "locus": ["model"] * 10 + ["framework"] * 10,
            "phase": ["action"] * 20,  # both axes uniform → undefined; sklearn returns 0
            "symptom": ["crash"] * 20,
            "root_cause": ["api_misuse"] * 20,
        },
    )
    report = kappa_report(primary=primary, secondary=secondary)
    # Locus: total disagreement on a 50/50 split → κ near -1.
    assert report["per_axis"]["locus"]["kappa"] < 0
    assert report["per_axis"]["locus"]["agreement_count"] == 0


def test_kappa_partial_agreement_yields_intermediate_score() -> None:
    """Realistic case: most labels match, a few don't."""
    nodes = [f"n{i}" for i in range(20)]
    primary = _label_df(
        nodes,
        {
            "locus": ["framework"] * 10 + ["model"] * 10,
            "phase": ["action"] * 20,
            "symptom": ["crash"] * 20,
            "root_cause": ["api_misuse"] * 20,
        },
    )
    # Flip 3 of 20 locus labels.
    flipped_locus = ["framework"] * 10 + ["model"] * 10
    flipped_locus[0] = "model"
    flipped_locus[1] = "model"
    flipped_locus[15] = "framework"
    secondary = _label_df(
        nodes,
        {
            "locus": flipped_locus,
            "phase": ["action"] * 20,
            "symptom": ["crash"] * 20,
            "root_cause": ["api_misuse"] * 20,
        },
    )
    report = kappa_report(primary=primary, secondary=secondary)
    # 17/20 matches on a 50/50 axis → κ around 0.7.
    assert 0.5 < report["per_axis"]["locus"]["kappa"] < 0.9
    assert report["per_axis"]["locus"]["agreement_count"] == 17


def test_kappa_inner_join_drops_unmatched_rows() -> None:
    """Only node_ids present in both tables contribute to κ."""
    primary = _label_df(
        ["n1", "n2", "n3"],
        {
            "locus": ["framework"] * 3,
            "phase": ["action"] * 3,
            "symptom": ["crash"] * 3,
            "root_cause": ["api_misuse"] * 3,
        },
    )
    secondary = _label_df(
        ["n2", "n3", "n4"],
        {
            "locus": ["framework"] * 3,
            "phase": ["action"] * 3,
            "symptom": ["crash"] * 3,
            "root_cause": ["api_misuse"] * 3,
        },
    )
    report = kappa_report(primary=primary, secondary=secondary)
    # n2 and n3 overlap; n1 and n4 are dropped.
    assert report["n"] == 2


def test_kappa_raises_on_empty_intersection() -> None:
    primary = _label_df(
        ["a"],
        {
            "locus": ["framework"],
            "phase": ["action"],
            "symptom": ["crash"],
            "root_cause": ["api_misuse"],
        },
    )
    secondary = _label_df(
        ["b"],
        {
            "locus": ["framework"],
            "phase": ["action"],
            "symptom": ["crash"],
            "root_cause": ["api_misuse"],
        },
    )
    with pytest.raises(RuntimeError, match="No overlapping"):
        kappa_report(primary=primary, secondary=secondary)


# --- Row → RawIssue conversion ------------------------------------------


def test_row_to_raw_issue_handles_none_body_and_labels() -> None:
    from datetime import UTC, datetime

    row = {
        "framework_slug": "x",
        "issue_number": 1,
        "node_id": "n1",
        "title": "t",
        "body": None,
        "url": "u",
        "labels": None,
        "state": "open",
        "is_pull_request": False,
        "created_at": datetime(2026, 4, 23, tzinfo=UTC),
        "updated_at": datetime(2026, 4, 23, tzinfo=UTC),
        "closed_at": None,
        "comment_count": 0,
    }
    raw = _row_to_raw_issue(row)
    assert raw.body is None
    assert raw.labels == ()
