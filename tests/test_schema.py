"""Schema round-trip tests — guard against accidental field renames
and enforce Literal constraints."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentfail.schema import (
    Classification,
    ClassifiedIssue,
    CrossLink,
    Framework,
    TaxonomyLabel,
)


def test_classification_rejects_unknown_locus():
    with pytest.raises(ValidationError):
        Classification(
            locus="totally_invented",  # type: ignore[arg-type]
            phase="action",
            symptom="crash",
            root_cause="api_misuse",
            confidence=0.9,
            reasoning="x",
        )


def test_classification_confidence_bounds():
    # 0.0 and 1.0 allowed
    Classification(
        locus="model",
        phase="action",
        symptom="wrong_output",
        root_cause="model_limitation",
        confidence=0.0,
        reasoning="x",
    )
    Classification(
        locus="model",
        phase="action",
        symptom="wrong_output",
        root_cause="model_limitation",
        confidence=1.0,
        reasoning="x",
    )
    # Out of range rejected
    with pytest.raises(ValidationError):
        Classification(
            locus="model",
            phase="action",
            symptom="wrong_output",
            root_cause="model_limitation",
            confidence=1.5,
            reasoning="x",
        )


def test_classified_issue_from_parts_roundtrip(raw_issue, classification_confident):
    row = ClassifiedIssue.from_parts(
        raw_issue,
        classification_confident,
        classifier_tier="haiku",
        classifier_model="claude-haiku-4-5-20251001",
        classifier_version="0.1.0",
        classified_at=datetime(2026, 4, 23, tzinfo=UTC),
    )
    assert row.framework_slug == raw_issue.framework_slug
    assert row.issue_number == raw_issue.issue_number
    assert row.locus == classification_confident.locus
    assert row.classifier_tier == "haiku"


def test_cross_link_requires_valid_corpus_id():
    CrossLink(
        node_id="abc",
        corpus_id="mast",
        external_id="mast-42",
        similarity=0.8,
        method="semantic",
    )
    with pytest.raises(ValidationError):
        CrossLink(
            node_id="abc",
            corpus_id="not_registered",  # type: ignore[arg-type]
            external_id="x",
            similarity=0.5,
            method="semantic",
        )


def test_framework_is_frozen():
    f = Framework(slug="x", repo="a/b", display_name="X")
    with pytest.raises(ValidationError):
        f.slug = "y"  # type: ignore[misc]


def test_taxonomy_label_axis_is_constrained():
    TaxonomyLabel(axis="locus", label="model", definition="x")
    with pytest.raises(ValidationError):
        TaxonomyLabel(axis="dimension_x", label="y", definition="z")  # type: ignore[arg-type]
