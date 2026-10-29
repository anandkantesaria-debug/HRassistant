from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langsmith.evaluation import evaluate
from pydantic import ValidationError

from eval.evaluators import build_evaluators
from eval.upload_dataset import (
    DEFAULT_DATASET_NAME,
    DEFAULT_GOLDEN_PATH,
    count_needs_review,
    load_golden_items,
)
from hrbot.config import Settings, setup_environment
from hrbot.graph import ask

DEFAULT_THRESHOLDS_PATH = Path("eval/thresholds.yaml")
DEFAULT_EVAL_LOG_PATH = Path("docs/eval_log.md")

AUTO_ROUTING_ALIASES = {
    "auto",
    "default",
    "router",
    "free",
    "combo",
    "any",
    "latest",
    "random",
    "unused",
}

METRIC_ORDER = [
    "retrieval_hit",
    "citation_present",
    "correct_escalation",
    "correctness",
    "groundedness",
]


@dataclass
class EvalRunSummary:
    experiment_prefix: str
    dataset_name: str
    metadata: dict[str, Any]
    mean_scores: dict[str, float]
    thresholds: dict[str, float]
    passed: bool
    failures: list[str]
    max_concurrency: int


def _resolve_settings(settings: Settings | None = None) -> Settings:
    if settings is not None:
        return settings
    try:
        return Settings()
    except ValidationError:
        return Settings(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY") or "unused",
            LLM_MODEL=os.environ.get("LLM_MODEL") or "unused",
        )


def is_explicit_model_id(model_id: str | None) -> bool:
    """Return True if `model_id` is a concrete model identifier and not an auto-routing alias."""
    if model_id is None:
        return False
    cleaned = model_id.strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if lower in AUTO_ROUTING_ALIASES:
        return False
    if lower.startswith(("auto/", "auto-", "combo/", "combo-", "router/", "router-")):
        return False
    return not lower.endswith(("/auto", "-auto", "/free", "/default"))


def validate_eval_models(settings: Settings) -> tuple[str, str]:
    """AC-7.5: Ensure both LLM_MODEL and EVAL_JUDGE_MODEL are explicit model IDs."""
    if not is_explicit_model_id(settings.LLM_MODEL):
        raise ValueError(
            f"LLM_MODEL ({settings.LLM_MODEL!r}) must be an explicit model ID for evaluation runs, "
            "not empty or an auto-routing alias."
        )
    if not is_explicit_model_id(settings.EVAL_JUDGE_MODEL):
        raise ValueError(
            f"EVAL_JUDGE_MODEL ({settings.EVAL_JUDGE_MODEL!r}) must be an explicit model ID for evaluation runs, "
            "not empty or an auto-routing alias."
        )
    return str(settings.LLM_MODEL), str(settings.EVAL_JUDGE_MODEL)


def build_experiment_metadata(settings: Settings) -> dict[str, Any]:
    """AC-7.5: Record generation model, judge model, chunk size, k, and threshold in metadata."""
    return {
        "llm_model": settings.LLM_MODEL,
        "eval_judge_model": settings.EVAL_JUDGE_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "k": settings.TOP_K,
        "top_k": settings.TOP_K,
        "threshold": settings.RELEVANCE_THRESHOLD,
        "relevance_threshold": settings.RELEVANCE_THRESHOLD,
    }


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, float]:
    """Load metric pass/fail thresholds from `eval/thresholds.yaml`."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {file_path}")
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in {file_path}")
    return {str(k): float(v) for k, v in raw.items()}


def make_target_fn(
    settings: Settings,
    *,
    llm: Any | None = None,
    retriever_fn: Any | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the target function passed to LangSmith `evaluate`."""

    def _target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", ""))
        result = ask(
            question,
            settings=settings,
            llm=llm,
            retriever_fn=retriever_fn,
        )
        return {
            "answer": result.answer,
            "route": result.route,
            "citations": [c.to_dict() for c in result.citations],
            "chunks": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": chunk.score,
                }
                for chunk in result.chunks
            ],
            "citations_missing": result.citations_missing,
            "error": result.error,
        }

    return _target


def extract_mean_scores(experiment_results: Any) -> dict[str, float]:
    """Compute mean score per evaluator key from LangSmith `evaluate` results (ignoring None scores)."""
    buckets: dict[str, list[float]] = {}

    if isinstance(experiment_results, dict) and all(
        isinstance(v, (int, float)) for v in experiment_results.values()
    ):
        return {str(k): float(v) for k, v in experiment_results.items()}

    for row in experiment_results:
        eval_container: Any = None
        if isinstance(row, dict):
            eval_container = row.get("evaluation_results")
        elif hasattr(row, "evaluation_results"):
            eval_container = row.evaluation_results

        if eval_container is None:
            continue

        if isinstance(eval_container, dict):
            results_list = eval_container.get("results") or []
        elif hasattr(eval_container, "results"):
            results_list = eval_container.results or []
        elif isinstance(eval_container, list):
            results_list = eval_container
        else:
            results_list = []

        for item in results_list:
            if isinstance(item, dict):
                key = item.get("key")
                score = item.get("score")
            else:
                key = getattr(item, "key", None)
                score = getattr(item, "score", None)

            if key and score is not None:
                buckets.setdefault(str(key), []).append(float(score))

    return {
        key: (sum(values) / len(values))
        for key, values in buckets.items()
        if len(values) > 0
    }


def check_thresholds(
    mean_scores: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, list[str]]:
    """Compare mean scores against thresholds and return (passed, failure_messages)."""
    failures: list[str] = []
    for metric, min_score in thresholds.items():
        actual = mean_scores.get(metric)
        if actual is None:
            failures.append(f"{metric}: missing score (threshold >= {min_score:.2f})")
        elif actual < min_score:
            failures.append(f"{metric}: {actual:.4f} < {min_score:.2f}")
    return len(failures) == 0, failures


def format_summary_table(
    mean_scores: dict[str, float],
    thresholds: dict[str, float],
) -> str:
    """Format a human-readable summary table of mean scores and threshold status."""
    all_metrics = list(METRIC_ORDER)
    for k in mean_scores:
        if k not in all_metrics:
            all_metrics.append(k)
    for k in thresholds:
        if k not in all_metrics:
            all_metrics.append(k)

    header = f"{'Metric':<22} | {'Mean Score':<12} | {'Threshold':<10} | {'Status':<6}"
    sep = "-" * len(header)
    lines = [header, sep]

    for metric in all_metrics:
        score_val = mean_scores.get(metric)
        thresh_val = thresholds.get(metric)
        score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
        thresh_str = f"{thresh_val:.2f}" if thresh_val is not None else "-"

        if thresh_val is None:
            status = "INFO"
        elif score_val is not None and score_val >= thresh_val:
            status = "PASS"
        else:
            status = "FAIL"

        lines.append(f"{metric:<22} | {score_str:<12} | {thresh_str:<10} | {status:<6}")

    return "\n".join(lines)


def append_eval_log(
    experiment_prefix: str,
    change_made: str,
    settings: Settings,
    mean_scores: dict[str, float],
    *,
    log_path: str | Path = DEFAULT_EVAL_LOG_PATH,
    run_date: str | None = None,
) -> str:
    """AC-7.7: Append a row to `docs/eval_log.md` with date, change made, config, and scores."""
    file_path = Path(log_path)
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "# Evaluation Log\n\n"
            "Records notable evaluation runs on `hr-policy-golden` (`eval/golden.jsonl`).\n\n"
            "| Date | Experiment | Change Made | Config (`LLM_MODEL`, `EVAL_JUDGE_MODEL`, `CHUNK_SIZE`, `TOP_K`, `RELEVANCE_THRESHOLD`) | `retrieval_hit` | `citation_present` | `correct_escalation` | `correctness` | `groundedness` |\n"
            "|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )

    date_str = run_date or dt.date.today().isoformat()
    config_str = (
        f"`llm={settings.LLM_MODEL}`, `judge={settings.EVAL_JUDGE_MODEL}`, "
        f"`chunk={settings.CHUNK_SIZE}`, `k={settings.TOP_K}`, "
        f"`threshold={settings.RELEVANCE_THRESHOLD}`"
    )

    def _fmt(key: str) -> str:
        val = mean_scores.get(key)
        return f"{val:.2f}" if val is not None else "N/A"

    row = (
        f"| {date_str} | `{experiment_prefix}` | {change_made} | {config_str} | "
        f"{_fmt('retrieval_hit')} | {_fmt('citation_present')} | "
        f"{_fmt('correct_escalation')} | {_fmt('correctness')} | {_fmt('groundedness')} |\n"
    )
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(row)
    return row


def run_evaluation(
    experiment_prefix: str,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    golden_path: str | Path = DEFAULT_GOLDEN_PATH,
    thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
    max_concurrency: int | None = None,
    log_change: str | None = None,
    log_path: str | Path = DEFAULT_EVAL_LOG_PATH,
    settings: Settings | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    llm: Any | None = None,
    judge_llm: Any | None = None,
    retriever_fn: Any | None = None,
) -> EvalRunSummary:
    """Run the LangGraph pipeline over the LangSmith dataset and score all 5 evaluators."""
    resolved_settings = _resolve_settings(settings)

    golden_file = Path(golden_path)
    if golden_file.exists():
        items = load_golden_items(golden_file)
        pending_review = count_needs_review(items)
        if pending_review > 0:
            raise RuntimeError(
                f"Refusing to run evaluation: {pending_review} row(s) in {golden_file} still have "
                "'needs_review': true. Verify golden.jsonl and remove 'needs_review' flags first."
            )

    validate_eval_models(resolved_settings)
    setup_environment(resolved_settings)

    thresholds = load_thresholds(thresholds_path)
    metadata = build_experiment_metadata(resolved_settings)
    concurrency = (
        max_concurrency
        if max_concurrency is not None
        else resolved_settings.EVAL_MAX_CONCURRENCY
    )
    if concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    target_fn = make_target_fn(
        resolved_settings,
        llm=llm,
        retriever_fn=retriever_fn,
    )
    evaluators = build_evaluators(
        settings=resolved_settings,
        judge_llm=judge_llm,
    )

    runner = evaluate_fn if evaluate_fn is not None else evaluate
    experiment_results = runner(
        target_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata=metadata,
        max_concurrency=concurrency,
    )

    mean_scores = extract_mean_scores(experiment_results)
    passed, failures = check_thresholds(mean_scores, thresholds)

    if log_change is not None:
        append_eval_log(
            experiment_prefix=experiment_prefix,
            change_made=log_change,
            settings=resolved_settings,
            mean_scores=mean_scores,
            log_path=log_path,
        )

    return EvalRunSummary(
        experiment_prefix=experiment_prefix,
        dataset_name=dataset_name,
        metadata=metadata,
        mean_scores=mean_scores,
        thresholds=thresholds,
        passed=passed,
        failures=failures,
        max_concurrency=concurrency,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation over the HR policy golden dataset."
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        required=True,
        help="Prefix for the LangSmith experiment name.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name (default: hr-policy-golden).",
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to local golden.jsonl file for review check (default: eval/golden.jsonl).",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
        help="Path to thresholds YAML file (default: eval/thresholds.yaml).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum evaluation concurrency (default: Settings.EVAL_MAX_CONCURRENCY = 2).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Explicit generation model ID override for this evaluation run.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Explicit judge model ID override for this evaluation run.",
    )
    parser.add_argument(
        "--log-change",
        type=str,
        default=None,
        help="Optional description of the change made; if set, appends a row to docs/eval_log.md.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_EVAL_LOG_PATH,
        help="Path to eval_log.md (default: docs/eval_log.md).",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    llm: Any | None = None,
    judge_llm: Any | None = None,
    retriever_fn: Any | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved_settings = _resolve_settings(settings)
    if args.llm_model or args.judge_model:
        updates: dict[str, Any] = {}
        if args.llm_model:
            updates["LLM_MODEL"] = args.llm_model
        if args.judge_model:
            updates["EVAL_JUDGE_MODEL"] = args.judge_model
        resolved_settings = resolved_settings.model_copy(update=updates)

    try:
        summary = run_evaluation(
            experiment_prefix=args.experiment_prefix,
            dataset_name=args.dataset_name,
            golden_path=args.golden_path,
            thresholds_path=args.thresholds,
            max_concurrency=args.max_concurrency,
            log_change=args.log_change,
            log_path=args.log_path,
            settings=resolved_settings,
            evaluate_fn=evaluate_fn,
            llm=llm,
            judge_llm=judge_llm,
            retriever_fn=retriever_fn,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Evaluation aborted: {exc}", file=sys.stderr)
        return 1

    print(f"Experiment prefix: {summary.experiment_prefix}")
    print(f"Dataset: {summary.dataset_name} (max_concurrency={summary.max_concurrency})")
    print(format_summary_table(summary.mean_scores, summary.thresholds))

    if not summary.passed:
        print("\nThreshold check FAILED:", file=sys.stderr)
        for failure in summary.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll threshold checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
