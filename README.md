# Agent Reliability Corpus

**A continuously-mined, cross-linked corpus of agent-framework failures.**

`agent-reliability-corpus` scrapes issues from the major LLM-agent framework
repositories, classifies each against a unified 4-axis failure taxonomy
derived from the academic literature, and publishes a versioned dataset to
the Hugging Face Hub every week. Each weekly snapshot is reproducible by
revision.

The project bundles two agents:

- **Agent A — pipeline** (`arc-pipeline`): stateless weekly job that
  runs `scrape → classify → cross-link → publish`.
- **Agent B — curator** (`arc-curator`): Pydantic-AI stateful agent
  that proposes new frameworks, audits classification drift, and tracks
  paper-worthy changes.

## Why this dataset exists

Agent-failure research is fragmented: five+ academic taxonomies have been
published in the last year (MAST, AgentFail, Agent Error Benchmark,
Characterizing Faults in Agentic AI, the 998-bug Framework Bugs study),
each on its own static snapshot, with no shared schema or cross-linking.
Researchers writing on agent reliability rebuild a partial corpus every
time. The Agent Reliability Corpus exists to be the canonical,
continuously-updated aggregation — grounded in real GitHub issues,
classified against a unified taxonomy, and cross-linked to the existing
academic records.

## Installation

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Quick tour

```python
from datasets import load_dataset

# Load the latest issues (streaming avoids downloading the whole corpus).
issues = load_dataset("USER/agent-reliability-corpus", "issues", split="train", streaming=True)

for row in issues.take(5):
    print(row["framework_slug"], row["locus"], row["symptom"], "-", row["title"])
```

Four configs are published; each is a standalone table on the Hub:

| config        | description                                                 |
| ------------- | ----------------------------------------------------------- |
| `issues`      | classified issues — one row per GitHub issue (primary)     |
| `cross_links` | links from issues to academic-corpus records (v1: sparse)   |
| `taxonomy`    | labels + definitions + which prior papers they derive from  |
| `frameworks`  | framework metadata (repo, homepage, slug)                   |

## Running the pipeline locally

```bash
export DEEPSEEK_API_KEY=...
export GITHUB_TOKEN=...   # optional but strongly recommended (5000 req/hr)

# Dry-run: no LLM calls, useful for smoke testing.
uv run arc-pipeline \
    --dry-run-classifier \
    --max-per-framework 5 \
    --output-dir data/snapshots

# Full run (pushes to HF; requires HF_TOKEN).
export HF_TOKEN=...
uv run arc-pipeline --push --hf-repo-id USER/agent-reliability-corpus
```

Weekly GitHub Actions pickup lives in `.github/workflows/weekly.yml`.

## Classifier design

Single-tier classifier using **DeepSeek V4-pro** via the OpenAI-compatible
API. One model call per issue, structured output enforced by tool calling
against the `Classification` Pydantic schema.

```
  RawIssue ──▶ deepseek-v4-pro (thinking disabled) ──▶ Classification
```

Reasoning ("thinking") mode is explicitly disabled. For a structured
tool-call task, reasoning tokens are pure cost overhead — the output is
a small, schema-constrained JSON object that doesn't benefit from inline
chain-of-thought, and DeepSeek bills reasoning tokens at output rates.

Every row records the model ID, classifier version, and timestamp, so each
label is fully reproducible. The ~3.6KB taxonomy block sits at the start
of every prompt and is byte-identical across calls; DeepSeek's automatic
prefix cache discounts it by ~50× on hit.

Estimated cost: well under $1 per 1,000 classified issues with cache-warm
runs. The classifier is built on Pydantic AI's per-provider model classes,
so swapping in another backend (Mistral, Together, OpenRouter, a local
vLLM server, any OpenAI-compatible endpoint) is a one-line change.

## Unified taxonomy

Four orthogonal axes, derived from the prior work:

- **locus** — where the failure manifests (model / agent / framework / workflow / platform)
- **phase** — which lifecycle stage (planning / action / reflection / memory / coordination / infra)
- **symptom** — observable manifestation (wrong_output / no_output / crash / loop / cost_overrun / security)
- **root_cause** — underlying reason (api_misuse / api_incompatibility / doc_desync / model_limitation / prompt_issue / data_quality / infrastructure)

See [`src/agentfail/taxonomy.py`](src/agentfail/taxonomy.py) and
[`DATASHEET.md`](DATASHEET.md) for full definitions and citation provenance.

## Citing this dataset

Once v1.0 is released with a DOI, cite as:

```bibtex
@dataset{agent_reliability_corpus_2026,
  title  = {Agent Reliability Corpus: A Living Cross-Linked Corpus of Agent Framework Failures},
  author = {Agent Reliability Corpus contributors},
  year   = {2026},
  url    = {https://huggingface.co/datasets/USER/agent-reliability-corpus},
}
```

## Contributing

- Proposals for new frameworks to include: open an issue with the repo and
  a one-paragraph rationale. The curator agent (Agent B) also auto-drafts
  proposals on its runs.
- Cross-corpus contributions (loaders for MAST / AgentFail / etc. records):
  see [`src/agentfail/crosslink.py`](src/agentfail/crosslink.py).

## License

- Code: [MIT](LICENSE)
- Dataset: [CC-BY-4.0](LICENSE-DATA)

The underlying GitHub issues remain under their respective project licenses;
this repository publishes derivative classifications + links, not the full
issue bodies verbatim where upstream licenses forbid it.

## Prior work cross-linked

- **MAST** — Multi-Agent System Failure Taxonomy
  ([arXiv 2503.13657](https://arxiv.org/abs/2503.13657))
- **AgentFail** — Lifecycle of Failures in Platform-Orchestrated Agentic Workflows
  ([arXiv 2509.23735](https://arxiv.org/abs/2509.23735))
- **Agent Error Benchmark** — Where LLM Agents Fail and How They Learn From Failures
  ([arXiv 2509.25370](https://arxiv.org/abs/2509.25370))
- **Characterizing Faults in Agentic AI**
  ([arXiv 2603.06847](https://arxiv.org/abs/2603.06847))
- **Framework Bugs Study** — Empirical Study of Bugs in Modern LLM Agent Frameworks
  ([arXiv 2602.21806](https://arxiv.org/abs/2602.21806))
