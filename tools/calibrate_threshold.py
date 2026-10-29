from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hrbot.retriever import RetrievedChunk, retrieve

DEFAULT_QUESTIONS_PATH = Path("eval/calibration_questions.jsonl")


@dataclass
class DistributionStats:
    count: int
    min: float
    p25: float
    median: float
    mean: float
    p75: float
    max: float


@dataclass
class CalibrationReport:
    answerable_scores: list[float]
    unanswerable_scores: list[float]
    answerable_stats: DistributionStats
    unanswerable_stats: DistributionStats
    suggested_threshold: float
    top5_source_hits: int
    top5_page_hits: int
    total_answerable: int


def load_calibration_questions(path: str | Path = DEFAULT_QUESTIONS_PATH) -> list[dict]:
    """Load calibration questions from a JSONL file."""
    file_path = Path(path)
    items: list[dict] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                items.append(json.loads(stripped))
    return items


def summarize_scores(scores: list[float]) -> DistributionStats:
    """Compute distribution summary statistics for a list of relevance scores."""
    if not scores:
        return DistributionStats(
            count=0, min=0.0, p25=0.0, median=0.0, mean=0.0, p75=0.0, max=0.0
        )
    arr = np.array(scores, dtype=float)
    return DistributionStats(
        count=len(scores),
        min=float(np.min(arr)),
        p25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        mean=float(statistics.fmean(scores)),
        p75=float(np.percentile(arr, 75)),
        max=float(np.max(arr)),
    )


def suggest_threshold(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
) -> float:
    """Suggest a RELEVANCE_THRESHOLD separating answerable and unanswerable scores."""
    if not answerable_scores or not unanswerable_scores:
        return 0.5

    min_ans = min(answerable_scores)
    max_unans = max(unanswerable_scores)

    if min_ans > max_unans:
        return round((max_unans + min_ans) / 2.0, 2)

    all_sorted = sorted(set(answerable_scores + unanswerable_scores))
    candidates: list[float] = []
    for i in range(len(all_sorted) - 1):
        candidates.append((all_sorted[i] + all_sorted[i + 1]) / 2.0)
    if not candidates:
        candidates = [all_sorted[0]]

    med_midpoint = (
        statistics.median(answerable_scores) + statistics.median(unanswerable_scores)
    ) / 2.0

    best_thresh = candidates[0]
    best_key = (-1.0, -1.0, float("-inf"))

    for thresh in candidates:
        tpr = sum(1 for s in answerable_scores if s >= thresh) / len(answerable_scores)
        tnr = sum(1 for s in unanswerable_scores if s < thresh) / len(unanswerable_scores)
        balanced_acc = (tpr + tnr) / 2.0
        key = (balanced_acc, tnr, -abs(thresh - med_midpoint))
        if key > best_key:
            best_key = key
            best_thresh = thresh

    return round(best_thresh, 2)


def run_calibration(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    k: int = 5,
    retrieve_fn: Callable[[str, int], list[RetrievedChunk]] | None = None,
) -> CalibrationReport:
    """Run retrieval over calibration questions, print distributions, and suggest threshold."""
    if retrieve_fn is None:
        def _default_retrieve(q: str, top_k: int) -> list[RetrievedChunk]:
            return retrieve(q, k=top_k)

        retrieve_fn = _default_retrieve

    questions = load_calibration_questions(questions_path)
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []
    top5_source_hits = 0
    top5_page_hits = 0
    total_answerable = 0

    print("=== Calibration Per-Question Results ===")
    for item in questions:
        q_text = str(item["question"])
        is_answerable = bool(item["answerable"])
        expected_source = item.get("expected_source")
        expected_page = item.get("expected_page")

        chunks = retrieve_fn(q_text, k)
        best_score = chunks[0].score if chunks else 0.0
        top_page = chunks[0].page if chunks else None

        if is_answerable:
            total_answerable += 1
            answerable_scores.append(best_score)
            source_hit = any(c.source == expected_source for c in chunks)
            page_hit = any(
                c.source == expected_source and c.page == expected_page for c in chunks
            )
            if source_hit:
                top5_source_hits += 1
            if page_hit:
                top5_page_hits += 1
            print(
                f"[ANSWERABLE]   best_score={best_score:.4f} (top_page={top_page}, "
                f"expected_page={expected_page}, top5_source_hit={source_hit}, "
                f"top5_page_hit={page_hit}) | {q_text}"
            )
        else:
            unanswerable_scores.append(best_score)
            print(
                f"[UNANSWERABLE] best_score={best_score:.4f} (top_page={top_page}) | {q_text}"
            )

    ans_stats = summarize_scores(answerable_scores)
    unans_stats = summarize_scores(unanswerable_scores)
    suggested = suggest_threshold(answerable_scores, unanswerable_scores)

    print("\n=== Best-Chunk Score Distribution ===")
    print(
        f"Answerable   (n={ans_stats.count}): "
        f"min={ans_stats.min:.4f}, p25={ans_stats.p25:.4f}, "
        f"median={ans_stats.median:.4f}, mean={ans_stats.mean:.4f}, "
        f"p75={ans_stats.p75:.4f}, max={ans_stats.max:.4f}"
    )
    print(
        f"Unanswerable (n={unans_stats.count}): "
        f"min={unans_stats.min:.4f}, p25={unans_stats.p25:.4f}, "
        f"median={unans_stats.median:.4f}, mean={unans_stats.mean:.4f}, "
        f"p75={unans_stats.p75:.4f}, max={unans_stats.max:.4f}"
    )
    print(
        f"\nTop-{k} source hit rate (answerable): {top5_source_hits}/{total_answerable}"
    )
    print(
        f"Top-{k} exact page hit rate (answerable): {top5_page_hits}/{total_answerable}"
    )
    print(f"Suggested RELEVANCE_THRESHOLD={suggested:.2f}")

    return CalibrationReport(
        answerable_scores=answerable_scores,
        unanswerable_scores=unanswerable_scores,
        answerable_stats=ans_stats,
        unanswerable_stats=unans_stats,
        suggested_threshold=suggested,
        top5_source_hits=top5_source_hits,
        top5_page_hits=top5_page_hits,
        total_answerable=total_answerable,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate RELEVANCE_THRESHOLD from eval/calibration_questions.jsonl."
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
        help="Path to calibration_questions.jsonl.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=5,
        help="Top-k chunks to retrieve per question (default: 5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_calibration(questions_path=args.questions, k=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())

