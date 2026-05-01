"""Cross-link our classified issues to records in academic prior corpora.

Architecture
------------
Each prior corpus is wrapped in a `RecordLoader` that returns a sequence
of `AcademicRecord` objects (a normalized {id, title, description}
shape). The `CrossLinker` then runs a TF-IDF + cosine-similarity match
between every ARC issue and every loaded record, emitting `CrossLink`
rows above a similarity threshold.

Why TF-IDF and not sentence embeddings:
    Bug reports and academic failure traces share lots of code-level
    terminology (framework names, function names, error strings) that
    exact-token matching catches reliably. Dense embeddings can
    over-generalize on this kind of input. v0 ships TF-IDF; a future
    version may layer in a dense-embedding pass for the lower
    similarity tail.

What v0 actually links
----------------------
Of the five prior corpora cross-referenced by the taxonomy, only MAST
publishes its underlying records as a structured public dataset
(`mcemri/MAST-Data` on Hugging Face). The other four are paper-only at
the time of writing. The architecture below registers all five and
implements a `RecordLoader` for MAST; the rest become drop-in once
their authors release structured data. The cross-link config on the
HF dataset is therefore non-empty (MAST hits) but sparse — explicitly
positioned in the paper as a v0 with a known growth path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agentfail.schema import ClassifiedIssue, CrossLink

log = structlog.get_logger(__name__)


# --- Records + loaders --------------------------------------------------


@dataclass(frozen=True)
class AcademicRecord:
    """One row from a prior academic corpus, normalized for matching.

    `text` is whatever string is most useful for similarity comparison —
    typically title + description concatenated. `corpus_id` and
    `external_id` together form the cross-link target.
    """

    corpus_id: str
    external_id: str
    title: str
    text: str


class RecordLoader(Protocol):
    """Loads records from a prior academic corpus.

    Concrete implementations are responsible for fetching whatever the
    upstream corpus publishes (HF dataset, GitHub repo, supplementary
    CSV) and normalizing each record into an `AcademicRecord`.
    """

    def load(self) -> Sequence[AcademicRecord]: ...


@dataclass(frozen=True)
class CorpusSpec:
    """Registry entry for an academic failure corpus."""

    corpus_id: str
    citation_key: str
    title: str
    url: str
    # None means: records are not yet ingested for this corpus. The
    # cross-linker simply skips it.
    records_loader: RecordLoader | None = None


class MASTLoader:
    """Loads MAST traces from `mcemri/MAST-Data` on Hugging Face Hub.

    MAST publishes two files:
      - MAD_human_labelled_dataset.json (~19 expert-annotated traces)
      - MAD_full_dataset.json (LLM-as-judge annotated, ~1K traces)

    We use the human-labeled file by default — higher-quality matches
    and small enough that the linker stays fast.
    """

    def __init__(self, *, prefer_full: bool = True) -> None:
        self._prefer_full = prefer_full

    def load(self) -> Sequence[AcademicRecord]:
        import json

        from huggingface_hub import hf_hub_download

        filename = (
            "MAD_full_dataset.json" if self._prefer_full else "MAD_human_labelled_dataset.json"
        )
        path = hf_hub_download(
            repo_id="mcemri/MAST-Data",
            filename=filename,
            repo_type="dataset",
        )
        with open(path) as f:
            raw = json.load(f)

        records: list[AcademicRecord] = []
        for r in raw:
            # MAST schema: round / mas_name / benchmark_name / trace_id /
            # trace / annotations. Use the trace text + the first
            # annotation's failure mode for matching.
            mas_name = r.get("mas_name", "")
            benchmark = r.get("benchmark_name", "")
            trace = r.get("trace", "")
            annotations = r.get("annotations") or []
            failure_modes = " | ".join(
                str(a.get("failure mode", "")).split("\n", 1)[0] for a in annotations
            )
            text = f"{mas_name} {benchmark} {failure_modes} {trace}".strip()
            external_id = f"{r.get('round', '')}-{mas_name}-{r.get('trace_id', '')}"
            records.append(
                AcademicRecord(
                    corpus_id="mast",
                    external_id=external_id,
                    title=f"MAST trace: {mas_name} on {benchmark}",
                    text=text,
                )
            )
        log.info("crosslink.mast_loaded", n=len(records), file=filename)
        return records


# --- Registry ----------------------------------------------------------

CORPORA: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        corpus_id="mast",
        citation_key="cemri2025mast",
        title="MAST: Multi-Agent System Failure Taxonomy",
        url="https://arxiv.org/abs/2503.13657",
        records_loader=MASTLoader(),
    ),
    CorpusSpec(
        corpus_id="agentfail",
        citation_key="agentfail2025",
        title="AgentFail: Lifecycle of Failures in Platform-Orchestrated Agentic Workflows",
        url="https://arxiv.org/abs/2509.23735",
        # No public structured dataset at time of writing. Loader pending.
    ),
    CorpusSpec(
        corpus_id="agent_error",
        citation_key="agenterror2025",
        title="Where LLM Agents Fail and How They Learn From Failures",
        url="https://arxiv.org/abs/2509.25370",
        # AgentErrorBench is on Google Drive (per the paper). Loader pending.
    ),
    CorpusSpec(
        corpus_id="faults_agentic",
        citation_key="faultsagentic2026",
        title="Characterizing Faults in Agentic AI",
        url="https://arxiv.org/abs/2603.06847",
        # Paper-only at time of writing. Loader pending.
    ),
    CorpusSpec(
        corpus_id="framework_bugs",
        citation_key="frameworkbugs2026",
        title="An Empirical Study of Bugs in Modern LLM Agent Frameworks",
        url="https://arxiv.org/abs/2602.21806",
        # Paper-only at time of writing. Loader pending.
    ),
)

CORPORA_BY_ID: dict[str, CorpusSpec] = {c.corpus_id: c for c in CORPORA}


# --- The linker --------------------------------------------------------


@dataclass
class CrossLinker:
    """Produces CrossLinks for classified issues against registered corpora.

    For every corpus with a configured loader, runs a TF-IDF + cosine
    pass against every classified issue and emits links above
    `threshold`. Issues that don't match any record in any corpus
    contribute zero links (the empty case is first-class for downstream
    code).
    """

    # 0.10 chosen empirically on a 14K-issue × 1.2K-MAST-record run:
    # at this threshold ~370 links survive, dominated by project-name
    # overlap (autogen ↔ Magentic, crewai ↔ MetaGPT). Higher thresholds
    # are publishable (15 links @ 0.15, 4 @ 0.20) but mask the long
    # tail of weaker-but-still-informative matches. Downstream users
    # can filter by `similarity` for tighter precision.
    threshold: float = 0.10
    corpora: tuple[CorpusSpec, ...] = field(default_factory=lambda: CORPORA)

    def link_batch(self, issues: Sequence[ClassifiedIssue]) -> list[CrossLink]:
        if not issues:
            return []

        # Collect records from every loader-equipped corpus.
        records: list[AcademicRecord] = []
        for spec in self.corpora:
            if spec.records_loader is None:
                log.info("crosslink.corpus_not_loaded", corpus_id=spec.corpus_id)
                continue
            try:
                loaded = list(spec.records_loader.load())
            except Exception as exc:
                log.warning(
                    "crosslink.loader_failed",
                    corpus_id=spec.corpus_id,
                    error=str(exc),
                )
                continue
            log.info("crosslink.corpus_loaded", corpus_id=spec.corpus_id, n=len(loaded))
            records.extend(loaded)

        if not records:
            return []

        # Render issues to text the same way the classifier sees them.
        issue_texts = [_issue_to_match_text(i) for i in issues]
        record_texts = [r.text for r in records]

        # TF-IDF over the union of issue + record corpora so the
        # vocabulary is shared. This makes cosine comparable across
        # corpora and avoids OOV blowups on technical terms.
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words="english",
            sublinear_tf=True,
        )
        all_texts = issue_texts + record_texts
        tfidf = vec.fit_transform(all_texts)
        issue_vecs = tfidf[: len(issue_texts)]
        record_vecs = tfidf[len(issue_texts) :]

        sim = cosine_similarity(issue_vecs, record_vecs)

        links: list[CrossLink] = []
        for i, issue in enumerate(issues):
            row = sim[i]
            for j, score in enumerate(row):
                if score < self.threshold:
                    continue
                rec = records[j]
                links.append(
                    CrossLink(
                        node_id=issue.node_id,
                        # mypy/pyright: corpus_id is a closed Literal; the
                        # registered loaders only emit valid values.
                        corpus_id=rec.corpus_id,  # type: ignore[arg-type]
                        external_id=rec.external_id,
                        similarity=float(score),
                        method="semantic",
                    )
                )
        log.info(
            "crosslink.batch_done",
            n_issues=len(issues),
            n_records=len(records),
            n_links=len(links),
            threshold=self.threshold,
        )
        return links


# --- Helpers -----------------------------------------------------------


def _issue_to_match_text(issue: ClassifiedIssue) -> str:
    """The blob of text we run TF-IDF over for one ARC issue.

    Same structure as the classifier's input, minus the per-issue
    metadata that would never overlap with academic record text.
    """
    body = (issue.body or "").strip()
    return f"{issue.framework_slug} {issue.title} {body}"
