# Datasheet for the Agent Reliability Corpus

Following the *Datasheets for Datasets* framework (Gebru et al., 2018).

## Motivation

**For what purpose was the dataset created?**
Research on agent-framework reliability is fragmented across several recent
academic corpora (MAST, AgentFail, Agent Error Benchmark, Characterizing
Faults in Agentic AI, Framework Bugs). Each is a static snapshot, none
cross-link, none continuously update. The Agent Reliability Corpus
provides a shared taxonomy and a continuously-maintained superset grounded
in real GitHub issues, so downstream researchers can work from one living
dataset rather than assembling per-paper ad-hoc corpora.

**Who created it and on behalf of whom?**
The Agent Reliability Corpus contributors, as an open-source community
project. No institutional funding.

## Composition

**What do instances represent?**
Each row in the primary `issues` config represents one GitHub issue
filed on an LLM agent-framework repository, classified against a unified
four-axis failure taxonomy.

**How many instances are there?**
Varies by snapshot; grows weekly. The initial release (v1.0) will cite
exact counts per framework.

**What data does each instance consist of?**

See `src/agentfail/schema.py::ClassifiedIssue`:

- Identifiers: `framework_slug`, `issue_number`, `node_id`, `url`
- Content: `title`, `body` (truncated to 4000 chars for classifier input;
  full body retained in the verbatim snapshot), `labels`, `state`
- Temporal: `created_at`, `updated_at`, `closed_at`
- Classification: `locus`, `phase`, `symptom`, `root_cause`, `confidence`,
  `reasoning`, `needs_review`
- Reproducibility: `classifier_tier`, `classifier_model`,
  `classifier_version`, `classified_at`

**Is there a label or target associated with each instance?**
Yes — four taxonomy labels (one per axis) plus a confidence score and a
boolean `needs_review` flag.

**Is any information missing?**
Issue bodies are sometimes empty (GitHub users filing title-only reports).
The `body` field is nullable; the classifier handles this and typically
sets `confidence < 0.6` + `needs_review = true` for such rows.

**Are there recommended data splits?**
Not in the traditional ML sense. Users commonly filter by:
- `framework_slug` for per-framework analyses
- `needs_review = false` for high-confidence subsets
- `classified_at >= DATE` for time-windowed studies
- `classifier_tier = "haiku"` to restrict to low-ambiguity cases

**Are there any errors, sources of noise, or redundancies?**

1. **Classifier noise.** Every LLM classifier has non-zero error. We
   record the model ID and classifier version per row, and confidence is
   reported on every classification. Monthly human audits sample ~50 rows
   for manual review (see `curator.py::audit_classification_drift`). The
   v0 classifier is single-tier (DeepSeek V4-pro); a multi-tier variant
   may be reintroduced if cost/quality tradeoffs warrant it.
2. **Non-failure issues.** The pre-filter is permissive; the classifier
   may receive feature requests or discussions and should return
   `symptom = unknown`, `needs_review = true` for those. Users wanting a
   stricter failure subset should filter `symptom != unknown`.
3. **Truncation.** Issue bodies > 4000 chars are truncated at classifier
   input; this may reduce context for very long issues. The truncation
   marker is visible in the classifier reasoning.
4. **Edge-of-taxonomy noise.** Issues that don't cleanly fit the
   four-axis taxonomy (feature requests, discussions that slipped the
   prefilter, ambiguous bug reports) tend to be labeled with `unknown`
   on one or more axes and `needs_review=true`. Filter on those fields
   for high-precision subsets.

**Is the dataset self-contained, or does it rely on external resources?**
Self-contained. The `cross_links` config references external academic
corpora by ID, but the corpus itself carries enough context (title, body,
URL) for most analyses.

## Collection Process

**How was the data acquired?**
GitHub REST API (`/repos/:owner/:repo/issues?state=all&sort=updated`),
paginated, authenticated via `GITHUB_TOKEN`. See `scrape.py`.

**What mechanisms were used?**
`httpx` + `tenacity` retry with exponential backoff. Rate-limit aware (the
scraper sleeps until the API's reset timestamp when throttled).

**Over what timeframe was the data collected?**
Continuous — each weekly run scrapes since the last successful update
(tracked per-framework in `data/pipeline_state.json`).

**Were any ethical review processes conducted?**
No IRB review. All data is public GitHub issue metadata, filed under
open-source projects' contribution guidelines. Personally identifying
information (usernames, email addresses) is not stored beyond what GitHub
already exposes on public issue pages. We do not scrape private
discussions or store comment threads beyond the opening report.

## Preprocessing / Cleaning / Labeling

**Was any preprocessing done?**
1. **Pre-filter.** Issues are dropped if they are pull requests or if
   their labels suggest a feature request / discussion and no failure
   signal is present in the title or labels. Filter logic is permissive;
   the LLM classifier is authoritative.
2. **Body truncation.** Classifier input is capped at 4000 chars to
   bound per-call cost. The stored `body` field retains the full text up
   to GitHub's native limit.
3. **Classification.** Every retained issue is labeled by the tiered LLM
   classifier; see README for the escalation logic.

**Is software used to clean/label available?**
Yes — this repository is that software.

## Uses

**Has the dataset been used for any tasks already?**
Initial release. Intended use cases:
- Empirical studies of agent-framework reliability
- Failure-type distribution analyses across frameworks and versions
- Training data for agent self-debugging (with care — see limitations)
- Benchmarks for automated failure-classification systems

**Is there a repository that links to known uses?**
See the GitHub repo's "Cited by" notes; and the Hugging Face dataset page
tracks dataset-level citations.

## Distribution

**How is the dataset distributed?**
Hugging Face Hub as a multi-config dataset. Each weekly run is a separate
git revision under the Hub repo. Loadable via `datasets.load_dataset()`.

**Is there a license?**
- Code: MIT
- Dataset: CC-BY-4.0

## Maintenance

**Who is maintaining the dataset?**
The Agent Reliability Corpus contributors. Contact via the GitHub repo.

**How often will it be updated?**
Weekly (Sunday 07:00 UTC) via GitHub Actions, so the snapshot is ready before Monday morning in every timezone. The curator agent (`agent B`)
runs on demand to propose schema and framework additions.

**Will older versions be retained?**
Yes. Every weekly snapshot is a separate revision on Hugging Face Hub; old
revisions remain accessible via their commit hash. This preserves
reproducibility of any paper that cites a specific revision.
