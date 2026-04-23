"""Taxonomy tests — protect prompt stability (cache hits) and label completeness."""

from __future__ import annotations

from agentfail.schema import Classification
from agentfail.taxonomy import (
    ALL_LABELS,
    LOCUS,
    PHASE,
    ROOT_CAUSE,
    SYMPTOM,
    render_taxonomy_for_prompt,
)


def test_taxonomy_prompt_is_deterministic():
    """Re-rendering must produce identical bytes — otherwise prompt caching
    invalidates between runs and we silently burn ~10x on cost."""
    a = render_taxonomy_for_prompt()
    b = render_taxonomy_for_prompt()
    assert a == b


def test_taxonomy_prompt_contains_every_label():
    prompt = render_taxonomy_for_prompt()
    for label in ALL_LABELS:
        assert f"**{label.label}**" in prompt, f"missing {label.axis}:{label.label}"


def test_taxonomy_literal_coverage_matches_schema():
    """Every Literal value in Classification must have a defining label and
    vice versa — otherwise the classifier could return a label with no
    definition in the prompt."""
    # Extract the Literal values from the Classification schema
    schema = Classification.model_json_schema()
    defs = schema.get("$defs", {})

    def literals_for(axis_name: str) -> set[str]:
        prop = schema["properties"][axis_name]
        if "enum" in prop:
            return set(prop["enum"])
        # If referenced via $ref, resolve
        if "$ref" in prop:
            key = prop["$ref"].split("/")[-1]
            return set(defs[key]["enum"])
        # Otherwise the values live in anyOf/oneOf
        raise AssertionError(f"can't extract literal values for {axis_name}")

    assert literals_for("locus") == {lbl.label for lbl in LOCUS}
    assert literals_for("phase") == {lbl.label for lbl in PHASE}
    assert literals_for("symptom") == {lbl.label for lbl in SYMPTOM}
    assert literals_for("root_cause") == {lbl.label for lbl in ROOT_CAUSE}


def test_every_label_has_a_nonempty_definition():
    for label in ALL_LABELS:
        assert label.definition.strip(), f"empty definition: {label.axis}:{label.label}"
