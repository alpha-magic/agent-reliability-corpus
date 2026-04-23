"""Cross-link our classified issues to records in the academic prior corpora.

Scope note (v0):
    The five academic corpora (MAST, AgentFail, Agent Error Benchmark,
    Characterizing Faults in Agentic AI, Framework Bugs) are not all
    publicly available as structured records at the time of writing —
    some ship only PDFs and supplementary CSVs. This module provides:

    1. A registry of corpora and their fetch strategies.
    2. A semantic-similarity linker stub that returns empty links when
       a corpus' records are not yet loaded. Paper-quality cross-linking
       is Phase 2 work (see project_agent_build.md).

    The public interface is stable: `CrossLinker.link_batch(issues)` always
    returns `Sequence[CrossLink]`. Downstream publisher code treats zero
    links as a first-class state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import structlog

from agentfail.schema import ClassifiedIssue, CrossLink

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CorpusSpec:
    """Registry entry for an academic failure corpus."""

    corpus_id: str
    citation_key: str
    title: str
    url: str
    # None means: records are not yet ingested for this corpus.
    records_loader: RecordLoader | None = None


class RecordLoader(Protocol):
    """Loads records from an academic corpus into a form the linker can match.

    Each corpus has its own supplementary data format; concrete implementations
    (added incrementally in Phase 2) will handle CSV, JSON, or structured
    PDF extraction.
    """

    def load(self) -> Sequence[AcademicRecord]: ...


@dataclass(frozen=True)
class AcademicRecord:
    """One row from an academic corpus, normalized enough to link against."""

    corpus_id: str
    external_id: str
    title: str
    description: str


# Registry. Entries without a loader will not produce cross-links in v0;
# adding a loader is a drop-in upgrade that doesn't touch the pipeline.
CORPORA: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        corpus_id="mast",
        citation_key="cemri2025mast",
        title="MAST: Multi-Agent System Failure Taxonomy",
        url="https://arxiv.org/abs/2503.13657",
    ),
    CorpusSpec(
        corpus_id="agentfail",
        citation_key="agentfail2025",
        title="AgentFail: Lifecycle of Failures in Platform-Orchestrated Agentic Workflows",
        url="https://arxiv.org/abs/2509.23735",
    ),
    CorpusSpec(
        corpus_id="agent_error",
        citation_key="agenterror2025",
        title="Where LLM Agents Fail and How They Learn From Failures",
        url="https://arxiv.org/abs/2509.25370",
    ),
    CorpusSpec(
        corpus_id="faults_agentic",
        citation_key="faultsagentic2026",
        title="Characterizing Faults in Agentic AI",
        url="https://arxiv.org/abs/2603.06847",
    ),
    CorpusSpec(
        corpus_id="framework_bugs",
        citation_key="frameworkbugs2026",
        title="An Empirical Study of Bugs in Modern LLM Agent Frameworks",
        url="https://arxiv.org/abs/2602.21806",
    ),
)

CORPORA_BY_ID: dict[str, CorpusSpec] = {c.corpus_id: c for c in CORPORA}


class CrossLinker:
    """Produces CrossLinks for classified issues against registered corpora.

    The linker is a stub in v0: until per-corpus RecordLoaders are
    implemented, `link_batch` returns `[]`. Downstream code (publish,
    tests, pipeline) must handle the empty case — keeping this side
    of the API exercised ensures the eventual live implementation
    drops in cleanly.
    """

    def __init__(self, corpora: tuple[CorpusSpec, ...] = CORPORA) -> None:
        self._corpora = corpora
        # Preload any corpus whose loader is configured
        self._records: dict[str, Sequence[AcademicRecord]] = {}
        for spec in corpora:
            if spec.records_loader is not None:
                self._records[spec.corpus_id] = spec.records_loader.load()
                log.info(
                    "crosslink.corpus_loaded",
                    corpus_id=spec.corpus_id,
                    n=len(self._records[spec.corpus_id]),
                )
            else:
                log.info("crosslink.corpus_not_loaded", corpus_id=spec.corpus_id)

    def link_batch(self, issues: Sequence[ClassifiedIssue]) -> list[CrossLink]:
        """Return all cross-links across all issues and all loaded corpora.

        In v0 this returns []. Phase-2 implementation will use sentence
        embeddings + threshold match per corpus_id.
        """
        if not self._records:
            return []
        # Phase 2: per-corpus semantic-similarity matching. Stub for now so
        # callers can't depend on the empty-list behavior.
        raise NotImplementedError(
            "Semantic cross-linking not yet implemented. Add RecordLoaders to CORPORA to enable."
        )
