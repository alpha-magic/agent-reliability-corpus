"""Tests for the cross-linker.

Mocks the academic-corpus loaders with synthetic records so no network
is hit. The shape and threshold logic of the TF-IDF + cosine pass is
what we want to lock in; the actual MAST loader is exercised
end-to-end by the post-backfill cross-link run (see PAPER.md).
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentfail.crosslink import (
    AcademicRecord,
    CorpusSpec,
    CrossLinker,
)
from agentfail.schema import ClassifiedIssue


def _issue(node_id: str, title: str, body: str = "") -> ClassifiedIssue:
    now = datetime(2026, 4, 23, tzinfo=UTC)
    return ClassifiedIssue(
        framework_slug="langchain",
        issue_number=1,
        node_id=node_id,
        title=title,
        body=body,
        url="https://example.com",
        labels=(),
        state="open",
        is_pull_request=False,
        created_at=now,
        updated_at=now,
        closed_at=None,
        comment_count=0,
        locus="framework",
        phase="action",
        symptom="crash",
        root_cause="api_misuse",
        confidence=0.9,
        reasoning="r",
        needs_review=False,
        classifier_tier="v4_pro",
        classifier_model="deepseek-v4-pro",
        classifier_version="0.1.0",
        classified_at=now,
    )


class _FakeLoader:
    def __init__(self, records: list[AcademicRecord]) -> None:
        self._records = records

    def load(self) -> list[AcademicRecord]:
        return self._records


class _ExplodingLoader:
    def load(self) -> list[AcademicRecord]:
        raise RuntimeError("upstream corpus unreachable")


def test_linker_emits_links_for_high_similarity_pairs() -> None:
    """Issue text and one record share enough vocabulary to clear the threshold."""
    issue = _issue(
        "n1",
        "ReAct agent infinite loop on tool retry",
        body=(
            "The ReAct planner gets stuck in a tool-call loop when the "
            "search tool throws on retry. Expected exponential backoff."
        ),
    )
    records = [
        AcademicRecord(
            corpus_id="mast",
            external_id="r-relevant",
            title="ReAct loop in tool retry",
            text=(
                "Trace: ReAct agent loops on tool retry, planner does not "
                "back off when search tool errors twice. Failure mode: "
                "infinite tool-call loop."
            ),
        ),
        AcademicRecord(
            corpus_id="mast",
            external_id="r-unrelated",
            title="Spotify playlist rating task",
            text="Give a 5-star rating to all songs in my Spotify playlist.",
        ),
    ]
    linker = CrossLinker(
        threshold=0.05,
        corpora=(
            CorpusSpec(
                corpus_id="mast",
                citation_key="cemri2025mast",
                title="MAST",
                url="https://arxiv.org/abs/2503.13657",
                records_loader=_FakeLoader(records),
            ),
        ),
    )
    links = linker.link_batch([issue])

    assert len(links) >= 1
    matched = [lk for lk in links if lk.external_id == "r-relevant"]
    assert matched, "the relevant record should clear the threshold"
    assert all(lk.node_id == "n1" for lk in links)
    assert all(lk.method == "semantic" for lk in links)
    assert all(0.0 <= lk.similarity <= 1.0 for lk in links)


def test_linker_returns_empty_when_no_loader_configured() -> None:
    """With every CorpusSpec lacking a loader, the linker is a no-op."""
    linker = CrossLinker(
        corpora=(
            CorpusSpec(
                corpus_id="mast",
                citation_key="cemri2025mast",
                title="MAST",
                url="https://arxiv.org/abs/2503.13657",
                records_loader=None,
            ),
        ),
    )
    assert linker.link_batch([_issue("n1", "x")]) == []


def test_linker_returns_empty_for_no_issues() -> None:
    """Empty input is a first-class case; linker must not crash."""
    linker = CrossLinker(corpora=())
    assert linker.link_batch([]) == []


def test_linker_skips_failing_loader() -> None:
    """A loader that raises should be logged and skipped, not bubble up."""
    issue = _issue("n1", "anything")
    linker = CrossLinker(
        corpora=(
            CorpusSpec(
                corpus_id="mast",
                citation_key="cemri2025mast",
                title="MAST",
                url="https://arxiv.org/abs/2503.13657",
                records_loader=_ExplodingLoader(),
            ),
        ),
    )
    # Should not raise; should return [] since the only loader failed.
    assert linker.link_batch([issue]) == []


def test_linker_threshold_filters_low_similarity_pairs() -> None:
    """Threshold gating: with the threshold raised above the actual
    similarity, no links should be emitted even when the loader is
    real."""
    issue = _issue("n1", "completely unrelated topic about pottery glazes")
    records = [
        AcademicRecord(
            corpus_id="mast",
            external_id="r1",
            title="ReAct agent loop bug",
            text="ReAct planner stuck retrying a search tool that 429s.",
        ),
    ]
    linker = CrossLinker(
        threshold=0.99,  # absurdly high, nothing should clear it
        corpora=(
            CorpusSpec(
                corpus_id="mast",
                citation_key="cemri2025mast",
                title="MAST",
                url="https://arxiv.org/abs/2503.13657",
                records_loader=_FakeLoader(records),
            ),
        ),
    )
    assert linker.link_batch([issue]) == []
