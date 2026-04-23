# agentfail

**A continuously-mined, cross-linked corpus of agent-framework failures.**

`agentfail` scrapes issues from the major LLM-agent framework repositories,
classifies each against a unified 4-axis failure taxonomy derived from the
academic literature, and publishes a versioned dataset to the Hugging Face
Hub every week. Each weekly snapshot is reproducible by revision.

The project bundles two agents:

- **Agent A — pipeline** (`agentfail.pipeline`): stateless weekly job that
  runs `scrape → classify → cross-link → publish`.
- **Agent B — curator** (`agentfail.curator`): Pydantic-AI stateful agent
  that proposes new frameworks, audits classification drift, and tracks
  paper-worthy changes.

## Why this dataset exists

Agent-failure research is fragmented: five+ academic taxonomies have been
published in the last year (MAST, AgentFail, Agent Error Benchmark,
Characterizing Faults in Agentic AI, the 998-bug Framework Bugs study),
each on its own static snapshot, with no shared schema or cross-linking.
Researchers writing on agent reliability rebuild a partial corpus every
time. `agentfail` exists to be the canonical, continuously-updated
aggregation — grounded in real GitHub issues, classified against a unified
taxonomy, and cross-linked to the existing academic records.

## Installation

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Quick tour

```python
from datasets import load_dataset

# Load the latest issues (streaming avoids downloading the whole corpus).
issues = load_dataset("USER/agentfail", "issues", split="train", streaming=True)

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
export ANTHROPIC_API_KEY=...
export GITHUB_TOKEN=...   # optional but strongly recommended (5000 req/hr)

# Dry-run: no Anthropic calls, useful for smoke testing.
uv run agentfail-pipeline \
    --dry-run-classifier \
    --max-per-framework 5 \
    --output-dir data/snapshots

# Full run (pushes to HF; requires HF_TOKEN).
export HF_TOKEN=...
uv run agentfail-pipeline --push --hf-repo-id USER/agentfail
```

Weekly GitHub Actions pickup lives in `.github/workflows/weekly.yml`.

## Classifier design

Issues flow through a tiered classifier with confidence-based escalation:

```
  Haiku 4.5  ─── high-confidence ───▶ accept
       │
       └─── low confidence ──▶ Sonnet 4.6 ─── high-confidence ───▶ accept
                                    │
                                    └─── low conf ──▶ Opus 4.7 ──▶ accept + flag
```

Every row records the tier, model ID, and classifier version, so each label
is fully reproducible. The taxonomy block (~3.6KB) is `cache_control`-marked
on the Anthropic API — the cached prefix cuts per-call cost by ~90% once
warmed.

Estimated cost: ~$10/month for ~10K classified issues/month with Haiku
dominance (~90% of calls).

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
@dataset{agentfail2026,
  title  = {agentfail: A Living Cross-Linked Corpus of Agent Framework Failures},
  author = {agentfail maintainers},
  year   = {2026},
  url    = {https://huggingface.co/datasets/USER/agentfail},
}
```

## Contributing

- Proposals for new frameworks to include: open an issue with the repo and
  a one-paragraph rationale. The curator agent (Agent B) also auto-drafts
  proposals on its runs.
- Cross-corpus contributions (loaders for MAST / AgentFail / etc. records):
  see [`src/agentfail/crosslink.py`](src/agentfail/crosslink.py).

## License

MIT for the code. The dataset itself is published under CC-BY-4.0 — the
underlying GitHub issues remain under their respective project licenses;
we publish derivative classifications + links, not the full issue bodies
verbatim where upstream licenses forbid it.

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
