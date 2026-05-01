"""Cross-model relabel + Cohen's κ for the inter-annotator validation step.

Two CLIs:

* ``arc-relabel`` — sample N issues from a published snapshot, run them
  through a *different* OpenAI-compatible classifier (Mistral, GPT-5,
  Gemini, …), and write a parquet of the resulting labels. The model is
  pure config: ``--model``, ``--base-url``, ``--api-key-env``,
  ``--extra-body`` flags.
* ``arc-kappa`` — given two labeled parquets (typically the original
  ``issues.parquet`` from the snapshot and the ``arc-relabel`` output),
  compute Cohen's κ per axis and emit a JSON report.

The intent is to make the cross-model agreement table that goes into
PAPER.md §6 cheap to regenerate against any provider, so the methodology
section can show that the labels are stable across model families
without locking into a specific second annotator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import structlog
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from sklearn.metrics import cohen_kappa_score
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentfail.classify import (
    DEEPSEEK_BASE_URL,
    MODEL_V4_PRO,
    Classifier,
)
from agentfail.schema import RawIssue

log = structlog.get_logger(__name__)

# The four taxonomy axes — each gets its own κ score in the report.
_AXES = ("locus", "phase", "symptom", "root_cause")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


# --- Sampling ----------------------------------------------------------


def stratified_sample(
    df: pl.DataFrame,
    *,
    n: int,
    stratify_by: str | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Sample n rows, optionally stratified by a column.

    Stratified mode guarantees a floor of 1 row per group (so even tiny
    groups appear) and distributes the remainder proportionally to
    group size, then adjusts so the total is exactly n. Without
    stratification, falls back to plain random sampling.
    """
    if n >= df.height:
        return df
    if stratify_by is None:
        return df.sample(n=n, seed=seed, shuffle=True)

    sizes = df.group_by(stratify_by).len().sort("len", descending=True)
    n_groups = sizes.height
    if n < n_groups:
        # Can't represent every group — fall back to plain random.
        log.warning(
            "stratified_sample.fallback_to_random",
            n=n,
            n_groups=n_groups,
            reason="n < n_groups",
        )
        return df.sample(n=n, seed=seed, shuffle=True)

    total = df.height
    # Step 1: floor-of-1 plus proportional remainder.
    targets: list[tuple[object, int, int]] = []  # (group_value, group_size, target)
    for row in sizes.rows(named=True):
        group_size = row["len"]
        proportional = round(n * group_size / total)
        target = max(1, min(group_size, proportional))
        targets.append((row[stratify_by], group_size, target))

    # Step 2: rebalance so sum(targets) == n exactly.
    def _total() -> int:
        return sum(t for _, _, t in targets)

    # Reduce the largest target until total <= n.
    while _total() > n:
        idx = max(range(len(targets)), key=lambda i: targets[i][2])
        if targets[idx][2] <= 1:
            break  # Can't go below floor; downstream concat just under-counts.
        g, s, t = targets[idx]
        targets[idx] = (g, s, t - 1)

    # Grow the largest under-allocated group until total == n.
    while _total() < n:
        candidates = [i for i, (_, s, t) in enumerate(targets) if s > t]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: targets[i][1] - targets[i][2])
        g, s, t = targets[idx]
        targets[idx] = (g, s, t + 1)

    out_parts: list[pl.DataFrame] = []
    for group_value, group_size, target in targets:
        if target <= 0:
            continue
        slice_df = df.filter(pl.col(stratify_by) == group_value).sample(
            n=min(target, group_size), seed=seed, shuffle=True
        )
        out_parts.append(slice_df)
    return pl.concat(out_parts, how="vertical_relaxed")


# --- Relabel ----------------------------------------------------------


def _row_to_raw_issue(row: dict) -> RawIssue:
    """Reconstruct a RawIssue from a `ClassifiedIssue` parquet row.

    Only the scrape-side fields are needed for relabel; we ignore the
    primary classifier's labels because we're producing fresh ones.
    """
    return RawIssue(
        framework_slug=row["framework_slug"],
        issue_number=row["issue_number"],
        node_id=row["node_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        labels=tuple(row["labels"]) if row["labels"] is not None else (),
        state=row["state"],
        is_pull_request=row["is_pull_request"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        closed_at=row["closed_at"],
        comment_count=row["comment_count"],
    )


def relabel(
    *,
    issues_parquet: Path,
    output_parquet: Path,
    classifier: Classifier,
    sample_size: int | None = None,
    stratify_by: str | None = "framework_slug",
    seed: int = 42,
) -> Path:
    """Run a fresh classifier pass over a sampled subset of an existing
    snapshot's issues. Returns the output path.

    Output columns: node_id, framework_slug, locus, phase, symptom,
    root_cause, confidence, reasoning, needs_review, classifier_model,
    classified_at. The `node_id` lets `arc-kappa` join against the
    primary labels.
    """
    df = pl.read_parquet(issues_parquet)
    log.info("relabel.loaded_issues", n=df.height, source=str(issues_parquet))

    if sample_size is not None and sample_size < df.height:
        df = stratified_sample(df, n=sample_size, stratify_by=stratify_by, seed=seed)
        log.info("relabel.sampled", n=df.height, stratify_by=stratify_by, seed=seed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=15),
        # Only retry on transient errors. 400-class errors (BadRequest,
        # UnprocessableEntity) come from schema-rejection of the model's
        # output and are deterministic — retrying just burns time. We hit
        # this on Llama 4 Scout against Groq's strict pre-validation
        # where a single failed issue cost ~60s of pointless retries.
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        ),
        reraise=True,
    )
    def _classify_with_retry(raw: RawIssue):
        return classifier._call(raw)  # noqa: SLF001 — intentionally using the internal call

    rows: list[dict] = []
    failures = 0
    for i, raw_row in enumerate(df.to_dicts()):
        node_id = raw_row["node_id"]
        try:
            raw_issue = _row_to_raw_issue(raw_row)
            cls = _classify_with_retry(raw_issue)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log.exception("relabel.classify_error", node_id=node_id, error=str(exc))
            continue
        rows.append(
            {
                "node_id": node_id,
                "framework_slug": raw_row["framework_slug"],
                "locus": cls.locus,
                "phase": cls.phase,
                "symptom": cls.symptom,
                "root_cause": cls.root_cause,
                "confidence": cls.confidence,
                "reasoning": cls.reasoning,
                "needs_review": cls.needs_review,
                "classifier_model": classifier._model,  # noqa: SLF001
                "classified_at": datetime.now(UTC),
            }
        )
        if (i + 1) % 100 == 0:
            log.info("relabel.progress", done=i + 1, total=df.height, failures=failures)

    out_df = pl.DataFrame(rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_parquet)
    log.info(
        "relabel.done",
        output=str(output_parquet),
        n=out_df.height,
        failures=failures,
        model=classifier._model,  # noqa: SLF001
    )
    return output_parquet


# --- Cohen's κ --------------------------------------------------------


def kappa_report(
    *,
    primary: pl.DataFrame,
    secondary: pl.DataFrame,
    axes: tuple[str, ...] = _AXES,
    join_on: str = "node_id",
) -> dict:
    """Compute Cohen's κ per axis between two label tables.

    Both inputs must contain `node_id` and the columns named in `axes`.
    Returns a dict suitable for JSON dumping with the per-axis κ, the
    raw agreement count, the per-axis confusion summary, and the sample
    size after the inner join.
    """
    joined = primary.select([join_on, *axes]).join(
        secondary.select([join_on, *axes]),
        on=join_on,
        how="inner",
        suffix="_b",
    )
    n_joined = joined.height
    if n_joined == 0:
        raise RuntimeError("No overlapping node_ids between primary and secondary tables.")

    per_axis: dict[str, dict] = {}
    for axis in axes:
        a = joined[axis].to_list()
        b = joined[f"{axis}_b"].to_list()
        kappa = float(cohen_kappa_score(a, b))
        agreement = sum(1 for x, y in zip(a, b, strict=True) if x == y)
        per_axis[axis] = {
            "kappa": kappa,
            "agreement_count": agreement,
            "agreement_rate": agreement / n_joined,
        }

    mean_kappa = sum(d["kappa"] for d in per_axis.values()) / len(per_axis)

    return {
        "n": n_joined,
        "axes": list(axes),
        "per_axis": per_axis,
        "mean_kappa": mean_kappa,
        "summary": {
            axis: f"κ={d['kappa']:.3f}, agreement={d['agreement_count']}/{n_joined}"
            for axis, d in per_axis.items()
        },
    }


# --- CLIs --------------------------------------------------------------


def relabel_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel a sampled subset of an ARC snapshot with an alternate "
            "classifier (different model and/or provider) — used as the "
            "second annotator for cross-model κ."
        )
    )
    parser.add_argument(
        "--issues-parquet",
        type=Path,
        required=True,
        help="Path to the snapshot's issues.parquet (the primary labels).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the relabel parquet.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_V4_PRO,
        help=(
            "Model ID for the alternate classifier. Examples: "
            "'mistral-medium-latest', 'gpt-5', 'gemini-2.5-flash'."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEEPSEEK_BASE_URL,
        help=(
            "OpenAI-compatible endpoint. Examples: "
            "'https://api.mistral.ai/v1', "
            "'https://api.openai.com/v1', "
            "'https://generativelanguage.googleapis.com/v1beta/openai/'."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help=(
            "Env var name to read the API key from. Examples: "
            "'MISTRAL_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY'."
        ),
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        help=(
            "JSON object to pass as the OpenAI SDK's extra_body. Default "
            'None. Use \'{"thinking":{"type":"disabled"}}\' for '
            "DeepSeek V4-pro to suppress reasoning tokens. Other "
            "providers don't accept unknown keys, so leave None for them."
        ),
    )
    parser.add_argument(
        "--use-max-completion-tokens",
        action="store_true",
        help=(
            "Use OpenAI's newer `max_completion_tokens` parameter instead "
            "of the legacy `max_tokens`. Required for GPT-5.x and o-series."
        ),
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help=(
            "Don't send `temperature` at all. Required for GPT-5.x and "
            "o-series, which only accept the default temperature."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help="How many issues to relabel. Use 0 to relabel all.",
    )
    parser.add_argument(
        "--stratify-by",
        default="framework_slug",
        help=("Column to stratify the sample on. Set to '' for plain random."),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    extra_body = json.loads(args.extra_body) if args.extra_body else None
    classifier = Classifier(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_body=extra_body,
        use_max_completion_tokens=args.use_max_completion_tokens,
        omit_temperature=args.omit_temperature,
    )
    relabel(
        issues_parquet=args.issues_parquet,
        output_parquet=args.output,
        classifier=classifier,
        sample_size=args.sample_size if args.sample_size > 0 else None,
        stratify_by=args.stratify_by or None,
        seed=args.seed,
    )


def kappa_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Cohen's κ per axis between two ARC label tables. "
            "Inputs are typically the snapshot's issues.parquet (primary) "
            "and an arc-relabel output (secondary)."
        )
    )
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="Path to the primary-labels parquet (typically issues.parquet).",
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        required=True,
        help="Path to the secondary-labels parquet (typically arc-relabel output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report. Always also prints to stdout.",
    )
    parser.add_argument(
        "--axes",
        default="locus,phase,symptom,root_cause",
        help="Comma-separated axes to compute κ for.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    primary = pl.read_parquet(args.primary)
    secondary = pl.read_parquet(args.secondary)
    report = kappa_report(primary=primary, secondary=secondary, axes=axes)

    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":  # pragma: no cover
    relabel_main()
