# Agent Reliability Corpus: A Continuously-Mined, Cross-Linked Dataset of LLM Agent-Framework Failures

**Status:** Draft v0.1 — for discussion with prospective co-authors.
**Code:** [github.com/alpha-magic/agent-reliability-corpus](https://github.com/alpha-magic/agent-reliability-corpus)
**Dataset:** [huggingface.co/datasets/mirotomasik/agent-reliability-corpus](https://huggingface.co/datasets/mirotomasik/agent-reliability-corpus)
**Licenses:** Code MIT, dataset CC-BY-4.0.

---

## Abstract

Research on LLM-agent reliability has produced at least five recent academic corpora — MAST [1], AgentFail [2], Agent Error Benchmark [3], Characterizing Faults in Agentic AI [4], and the Framework Bugs Study [5] — each with its own taxonomy and a one-shot snapshot of failures. Cross-corpus comparison and longitudinal analysis are nearly impossible: the taxonomies don't align, the records don't link, and every follow-up paper rebuilds its evaluation set from scratch. We present the **Agent Reliability Corpus (ARC)**, a continuously-mined dataset of 14,129 classified GitHub issues from 12 LLM-agent frameworks, labeled against a unified four-axis taxonomy (*locus*, *phase*, *symptom*, *root cause*) synthesized from the five prior corpora with explicit derivation provenance on every label. ARC ships full reproducibility metadata on every row, costs ~$0.0002 per classification thanks to provider-side prefix caching, and includes a TF-IDF cross-link table from each ARC issue to records in MAST. A weekly fan-out-then-merge GitHub Actions workflow extends the corpus and publishes a new versioned revision to the Hugging Face Hub. We are seeking a co-author for the inter-annotator validation step described in §6.

---

## 1. Introduction

The empirical study of LLM-agent reliability has accelerated rapidly since 2025. Within twelve months we have seen five substantial corpora published (Table 1), each making careful empirical claims about how, why, and where agents fail. Yet anyone working on a follow-up paper today faces a now-familiar problem: the corpora are mutually inarticulate. MAST's *system design* category does not map cleanly onto AgentFail's *coordination* axis or the Framework Bugs Study's *Self-Action* lifecycle stage. Records in one corpus are not linked to records in another. None of them update — every dataset is a snapshot of the field at the moment its authors finished labeling.

| Corpus | Year | Source | Records | Taxonomy depth |
|---|---|---|---|---|
| MAST [1] | 2025 | 7 MAS frameworks, 1.6K traces | 14 modes / 3 categories | trace-level |
| AgentFail [2] | 2025 | Dify, Coze platform logs | 307 cases | lifecycle stages |
| AgentErrorBench [3] | 2025 | ALFWorld, GAIA, WebShop | 200 trajectories | 5 axes |
| Characterizing Faults [4] | 2026 | mixed agentic systems | n/a | type/symptom/cause |
| Framework Bugs Study [5] | 2026 | CrewAI, LangChain | 998 issues | 5 lifecycle stages |

*Table 1: Recent academic corpora on LLM-agent failures. None update; none cross-link.*

The Agent Reliability Corpus exists to be the canonical, continuously-updated, cross-linked aggregation. Its contribution is not a sixth taxonomy — it is **the dataset that connects them all**, kept fresh by a weekly pipeline that costs cents to run.

**Contributions.**

1. A unified 4-axis 28-label taxonomy synthesized from the five prior corpora, with each label tagged by which prior paper it derives from (the `derived_from` column on `taxonomy.parquet`).
2. 14,129 classified GitHub issues across 12 LLM-agent frameworks (`agno`, `autogen`, `autogpt`, `crewai`, `langchain`, `langgraph`, `letta`, `llamaindex`, `mastra`, `semantic_kernel`, `smolagents`, `swarm`), with full reproducibility metadata (pinned model ID, classifier version, classification timestamp) on every row.
3. A TF-IDF + cosine cross-link table from ARC issues to records in MAST (the only prior corpus that publishes its underlying records as a structured public dataset at time of writing). 368 links above similarity 0.10, dominated by genuine project-name overlap (e.g. autogen issues against MAST's Magentic-One traces).
4. A reproducible end-to-end pipeline that runs at ~$0.0002 per classification on DeepSeek V4-pro, with the whole 14K-issue backfill costing ~$3. The weekly cron extends the corpus to a new revision branch each Monday at 07:00 UTC.

---

## 2. Related work

The five prior corpora the unified taxonomy derives from:

- **MAST** [1] introduced the first multi-agent system failure taxonomy (3 categories, 14 modes) with κ = 0.88 inter-annotator agreement on 150 traces, validated against 1,600 LLM-judge-annotated traces from 7 MAS frameworks.
- **AgentFail** [2] (Demystifying the Lifecycle of Failures in Platform-Orchestrated Agentic Workflows) studied 307 real-world failure cases from Dify and Coze, focusing on visual-workflow platforms rather than code libraries.
- **Agent Error Benchmark / AgentDebug** [3] introduced AgentErrorTaxonomy (memory / reflection / planning / action / system) and a 200-trajectory benchmark from ALFWorld, GAIA, and WebShop, plus a debugging framework that recovers 26% of failures.
- **Characterizing Faults in Agentic AI** [4] proposed a 3-axis (type / symptom / root cause) taxonomy across mixed agentic systems.
- **Framework Bugs Study** [5] examined 998 bug reports from CrewAI and LangChain, identifying 15 root causes and 7 symptoms across five lifecycle stages.

ARC's taxonomy synthesizes these into four orthogonal axes (§4) and credits each prior paper via `derived_from` provenance on every label, so a researcher loading ARC can filter to "rows whose `locus` originates in MAST" or compose mappings back to any single prior taxonomy.

Adjacent work includes [6] (Dissecting Bug Triggers in Modern Agentic Frameworks; 409 bugs across 5 frameworks), and benchmarking work like [7] (MultiAgentBench). These do not currently feed into the cross-link table but are obvious near-term additions.

---

## 3. Dataset construction

### 3.1 Frameworks

We scrape GitHub issues from the twelve open-source repositories that together account for the bulk of the LLM-agent ecosystem at time of writing: `agno-agi/agno`, `microsoft/autogen`, `Significant-Gravitas/AutoGPT`, `crewAIInc/crewAI`, `langchain-ai/langchain`, `langchain-ai/langgraph`, `letta-ai/letta`, `run-llama/llama_index`, `mastra-ai/mastra`, `microsoft/semantic-kernel`, `huggingface/smolagents`, and `openai/swarm`. The list deliberately mixes Python and TypeScript projects, library-style and orchestration-style frameworks, large established codebases (langchain, autogpt) and newer entrants (mastra, agno). Adding a framework is a one-line change in `frameworks.py`; we expect the Curator agent (Agent B; see §3.5) to propose new entries on every weekly run.

### 3.2 Scraping

Each framework's `/repos/owner/repo/issues?state=all&sort=updated` endpoint is paginated through with an authenticated GitHub token (5,000 requests/hour). A permissive heuristic prefilter drops obvious non-failures (open pull requests, issues whose labels include only `enhancement` / `discussion` and whose body lacks failure keywords) but errs toward the LLM classifier — false positives at this stage are caught and labeled `symptom = unknown`, `needs_review = true` downstream.

The pipeline is *incremental by default*: a small `pipeline_state.json` records the most recent `updated_at` timestamp seen per framework, and subsequent runs scrape only deltas. The full backfill described here was the cold-start case; weekly runs each touch a few hundred issues at most.

### 3.3 Classifier

The classifier is single-tier: every issue is labeled by **DeepSeek V4-pro** [8] via the OpenAI-compatible Chat Completions API, with structured output enforced by tool calling against a Pydantic schema covering all four axes plus a confidence score, free-form one-sentence reasoning, and a `needs_review` boolean. The model is invoked with `temperature=0` and `thinking={"type": "disabled"}` — for a structured tool-call task on a closed-Literal output schema, reasoning tokens are pure cost overhead and do not improve label quality.

The system message places the ~3.6 KB taxonomy block at the start of every prompt. DeepSeek's automatic prefix cache detects this byte-identical prefix and bills cached input at $0.003625/M instead of $0.435/M (the current 75% discounted rates valid through 2026-05-31), yielding a 97% cache-hit ratio on the v0 backfill and a measured **cost of $0.000212 per classification**.

The choice to use DeepSeek V4-pro rather than a frontier closed model deserves comment. We previously evaluated a tiered Anthropic stack (Haiku 4.5 → Sonnet 4.6 → Opus 4.7 with confidence-based escalation) and observed 54% of issues escalating all the way to Opus due to needs_review-triggered escalation, yielding a per-issue cost of ~$0.014. Switching to single-tier V4-pro produced *higher* mean confidence (0.84 vs 0.70) and *lower* cost simultaneously — DeepSeek's training mix on technical / code-adjacent text appears well-suited to bug classification. We argue elsewhere that DeepSeek V4-pro is now a defensible choice for academic dataset labeling: the cost story makes the dataset *more* reproducible by follow-up researchers, not less, and DeepSeek's published model snapshots are pinnable to the same standard as Anthropic's.

The classifier component is deliberately swappable: the implementation uses the OpenAI SDK with a custom `base_url`, so any OpenAI-compatible backend (Mistral, Together, OpenRouter, a local vLLM server) is a one-line change. Future revisions of the corpus can use a different classifier without changing any other code.

### 3.4 Cross-linking

After classification, each issue is compared against records from any prior corpus that publishes a structured public dataset. At time of writing only **MAST** does so [`mcemri/MAST-Data` on Hugging Face Hub]; the other four prior corpora are paper-only and remain registered in `CORPORA` with `loader=None` placeholders. We expect this to change as datasheet expectations harden across the field.

The matcher is TF-IDF + cosine similarity, computed over the union of ARC issue text and academic record text so the vocabulary is shared. The default threshold is 0.10. The choice was empirical — at 0.10 the corpus yields 368 links across 1,242 MAST traces and the 14,129 ARC issues, dominated by project-name overlap (10 of the top 15 highest-similarity matches are autogen issues mentioning `MultimodalWebSurfer` matching MAST traces from the Magentic-One project). Higher thresholds (0.15 → 15 links, 0.20 → 4 links) are publishable but mask the long tail of weaker-but-still-informative matches; downstream users can filter by `similarity` for tighter precision. We make no claim that TF-IDF is the optimal matcher; replacing it with sentence embeddings is a one-class change and will likely be a v0.2 enhancement.

### 3.5 Curator agent

A second agent (Agent B, the Curator) runs on demand (not on the weekly cron) and proposes new frameworks to scrape, audits classification drift on a sampled subset, and assesses paper-worthiness of accumulated changes. The Curator is built on Pydantic AI with the same DeepSeek V4-pro backend; every action it takes goes into `actions_requiring_human_approval` rather than executing directly. It is a quality-of-life tool, not part of the citation surface.

---

## 4. The unified taxonomy

ARC labels every issue along four orthogonal axes:

- **Locus** (where the failure manifests): `model` | `agent` | `framework` | `workflow` | `platform` | `unknown`
- **Phase** (which lifecycle stage): `planning` | `action` | `reflection` | `memory` | `coordination` | `infra` | `unknown`
- **Symptom** (observable manifestation): `wrong_output` | `no_output` | `cost_overrun` | `crash` | `security` | `loop` | `unknown`
- **Root cause** (underlying reason): `api_misuse` | `api_incompatibility` | `doc_desync` | `model_limitation` | `prompt_issue` | `data_quality` | `infrastructure` | `unknown`

This is 28 labels in total. Each label carries a `derived_from` tuple naming which prior corpora the label traces back to (e.g. `derived_from = ("agent_error", "agentfail")`). The full table — 4 axes × labels × definitions × `derived_from` provenance — ships as the `taxonomy` config of the published dataset.

The explicit `unknown` per axis is load-bearing: it lets the classifier admit when an issue is off-taxonomy (a feature request that slipped the prefilter, a discussion thread, an ambiguous bug report) without forcing a guess. Approximately 40% of issues have at least one `unknown` axis label, concentrated in `root_cause` (44%) and `symptom` (39%); these rows are usually flagged with `needs_review = true` and are the natural input to Curator drift audits.

---

## 5. Cross-linking — example matches

A handful of the highest-similarity TF-IDF matches as illustrative anecdotes. None of these have been verified manually — verification is part of the inter-annotator validation in §6.

1. **autogen issue: "When Call MultimodalWebSurfer after using tool, error occurred"** ↔ **MAST trace `Magentic-146`** (similarity 0.202). Both texts describe a failure in the Magentic-One MultimodalWebSurfer component; the MAST trace is a synthetic execution of the same component, the ARC issue is a real bug report against it.
2. **autogen issue: "Magentic-One agents assume Linux shell on Windows — subprocess commands fail"** ↔ **MAST `Magentic-146`** (0.174). A platform-specific bug not present in the MAST trace itself, but the trace mentions the same subprocess call sites.
3. **crewai issue: "Agent responses are getting cut-off very time"** ↔ **MAST `MetaGPT-6`** (0.188). MetaGPT and CrewAI are different MAS frameworks; both texts describe truncation symptoms in agent-to-agent communication.

These are exactly the kind of cross-corpus links that the dataset is built to enable. A researcher looking at MAST trace `Magentic-146` can now find ten real-world ARC issues that exhibit related failure modes; conversely, a researcher triaging the autogen issue queue can see which Magentic-One bugs have been studied as benchmark traces.

---

## 6. Empirical analysis and validation roadmap

### 6.1 v0 corpus characteristics

The 14,129 issues in the published v0 break down as follows:

| Framework | Issues | Note |
|---|---|---|
| autogpt | 1,500 | hit per-framework cap |
| semantic_kernel | 1,499 | hit cap |
| langchain | 1,499 | hit cap |
| llamaindex | 1,498 | hit cap |
| agno | 1,498 | hit cap |
| mastra | 1,495 | hit cap |
| autogen | 1,441 | hit cap (some prefilter drops) |
| crewai | 1,229 | nearly full coverage |
| langgraph | 1,050 | full coverage |
| letta | 765 | full coverage |
| smolagents | 628 | full coverage |
| swarm | 27 | full coverage (archived repo) |
| **Total** | **14,129** | |

Confidence distribution: mean **0.84**, median **0.85**, with **87%** of issues at high confidence (≥0.8) and only ~1% at low confidence (<0.5). 12% of issues are flagged `needs_review = true`, a function the Curator uses to sample drift-audit candidates.

Per-axis label distribution (full counts in `issues.parquet`):
- **Locus:** dominated by `framework` (67%), then `unknown` (22%), `platform` (8%), `agent` (5%).
- **Symptom:** most common are `crash` (36%) and `wrong_output` (28%), with `unknown` at 23%.
- **Root cause:** `api_misuse` (52%) is the dominant pattern — consistent with the Framework Bugs Study's findings on CrewAI and LangChain — followed by `unknown` (27%) and `api_incompatibility` (9%).

These distributions are descriptive, not prescriptive. We do not yet have the data to claim that, say, "62% of LangChain failures are root-caused in API misuse" — that would require a held-out gold standard.

### 6.2 The validation gap

The v0 corpus is **labeled by a single LLM classifier** (DeepSeek V4-pro). For the dataset to be peer-review-ready, two validations are required:

**Inter-annotator agreement (κ).** A 200-row stratified subset (sampled across frameworks, with oversampling of `needs_review = true` rows) needs to be labeled independently by two human annotators against the four-axis taxonomy. Cohen's κ is reported per axis. The bar for publication is mean κ ≥ 0.7 ("substantial" agreement); MAST achieved κ = 0.88 on its 150-trace sample. **This is the load-bearing piece of work that gates submission and is the natural co-author contribution.** The labeling tooling is already scaffolded; the tasks are ~6 hours of focused work per annotator.

**Classifier vs gold standard.** Once the κ subset exists, we report DeepSeek V4-pro's accuracy against the human gold standard. We expect the dominant disagreement modes to be: (a) machine over-uses `unknown` on borderline cases that humans resolve into specific labels, and (b) on ambiguous root_cause judgments where the issue body lacks decisive evidence, humans converge on prior-knowledge inference while the classifier stays conservative. The accuracy decomposition gives reviewers a clear sense of where machine labels are trustworthy and where they aren't.

These two pieces are explicitly part of the v1.0 release plan. The published v0 dataset includes a clearly-labeled "unverified labels" caveat on the dataset card.

---

## 7. Limitations

- **Sparse cross-links.** Only one of the five prior corpora (MAST) currently publishes structured records. The cross-link table will grow as the others release data; the architecture is in place.
- **Single classifier; pending κ.** As above (§6.2).
- **English bias.** Issue bodies are not translated; non-English issues classify with reduced accuracy. The framework selection is also Anglophone-skewed.
- **TF-IDF is a baseline.** A v0.2 cross-linker using sentence embeddings is straightforward and likely improves the long-tail recall without sacrificing precision.
- **Frameworks selection.** The 12 frameworks cover the bulk of the open-source ecosystem but exclude proprietary stacks (e.g. closed corporate agents) and rapid newcomers. The Curator agent's `propose_new_framework` tool is the maintenance mechanism.
- **GitHub-only data source.** Issues filed elsewhere (Discord, vendor support tickets, internal incident systems) are not covered. Open-source issues are a representative but not exhaustive sample of agent-framework failure.
- **The cross-corpus mapping is a synthesis, not a reduction.** Some original distinctions in the prior taxonomies are lost or merged in the unified 4-axis space. The `derived_from` provenance on each label is the audit trail; researchers wanting the original distinction can compose it back.

---

## 8. Reproducibility

Every classified row records its `classifier_model` (`deepseek-v4-pro`), `classifier_version` (the ARC release tag that produced the label), and `classified_at` (UTC ISO timestamp). Every weekly snapshot is published to the Hugging Face Hub as both a date-stamped revision branch (`2026-W18`, `2026-W19`, …) and an updated `main` branch. Papers should cite a specific revision:

```python
load_dataset("mirotomasik/agent-reliability-corpus", "issues", revision="2026-W18")
```

Code, taxonomy, prompt template, threshold parameters, and pipeline state are all in the public repository with an MIT license. The dataset itself is CC-BY-4.0. The classifier system prompt is reproduced in full in `src/agentfail/classify.py` and `taxonomy.py::render_taxonomy_for_prompt()`.

The full v0 backfill — all 14,129 classifications — was reproduced from scratch in **~5 hours wall-clock** by a 12-way GitHub Actions matrix at a total API cost of **~$3.00**. The reproducibility floor is therefore "$3 and one workday" for any researcher with a DeepSeek API key.

---

## 9. Conclusion and call for collaboration

The Agent Reliability Corpus is built. It is on Hugging Face Hub. It is licensed permissively. The cross-link table is populated against MAST. The pipeline runs weekly and costs cents.

What is missing is the inter-annotator validation that turns this from a credible engineering artifact into a publishable academic dataset. We are looking for one collaborator — ideally a researcher whose prior or upcoming work makes use of one or more of the corpora ARC cross-links against — to (i) co-label a 200-row subset for κ scoring, (ii) co-write the methodology and validation sections of a workshop submission, and (iii) be a co-first or second author on the resulting paper. The work is bounded, the dataset is real, and the venue (NeurIPS Datasets & Benchmarks track, or a relevant agent workshop) is concrete.

If you've published one of [1]–[5], or if you cite multiple of them in your own pipeline, this dataset should already feel useful to you. Email or open an issue on the GitHub repository to discuss.

---

## References

[1] Cemri, M., Pan, M. Z., Yang, S., et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657. [https://arxiv.org/abs/2503.13657](https://arxiv.org/abs/2503.13657)

[2] Ma, X., Wang, Y., Wang, J., Wang, Q. (2025). *Demystifying the Lifecycle of Failures in Platform-Orchestrated Agentic Workflows.* arXiv:2509.23735. [https://arxiv.org/abs/2509.23735](https://arxiv.org/abs/2509.23735)

[3] Zhu, K., Liu, Z., Li, B., et al. (2025). *Where LLM Agents Fail and How They Can Learn From Failures.* arXiv:2509.25370. [https://arxiv.org/abs/2509.25370](https://arxiv.org/abs/2509.25370)

[4] Shah, M. B., Morovati, M. M., Rahman, M. M., Khomh, F. (2026). *Characterizing Faults in Agentic AI: A Taxonomy of Types, Symptoms, and Root Causes.* arXiv:2603.06847. [https://arxiv.org/abs/2603.06847](https://arxiv.org/abs/2603.06847)

[5] Zhu, X., Wu, J., Zhang, X., et al. (2026). *An Empirical Study of Bugs in Modern LLM Agent Frameworks.* arXiv:2602.21806. [https://arxiv.org/abs/2602.21806](https://arxiv.org/abs/2602.21806)

[6] Zhang, X., Zhang, H., Tan, S. H. (2026). *Dissecting Bug Triggers and Failure Modes in Modern Agentic Frameworks: An Empirical Study.* arXiv:2604.08906. [https://arxiv.org/abs/2604.08906](https://arxiv.org/abs/2604.08906)

[7] Zhu, K., et al. (2025). *MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents.* arXiv:2503.01935 (ICML 2025).

[8] DeepSeek. (2026). *DeepSeek V4 Technical Report.* (model documentation).
