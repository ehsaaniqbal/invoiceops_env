#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

TASK_ORDER = ("easy", "medium", "medium_plus", "hard")


def find_latest_eval() -> Path:
    candidates = sorted(Path("outputs/evals").glob("*.json"))
    if not candidates:
        raise FileNotFoundError("No eval JSON files found under outputs/evals/.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact summary for an InvoiceOps eval JSON artifact."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional eval JSON paths. Defaults to the latest file under outputs/evals/.",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _request_error_count(result: dict[str, Any]) -> int:
    attempts = result.get("model_attempts") or []
    return sum(
        1
        for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("request_error")
    )


def summarize_eval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") or []

    task_scores: dict[str, float] = {}
    resolution_scores: list[float] = []
    evidence_scores: list[float] = []
    documentation_scores: list[float] = []
    efficiency_scores: list[float] = []
    steps: list[float] = []
    reward_lengths: list[float] = []
    fallback_count = 0
    parse_failure_count = 0
    request_error_count = 0

    for result in results:
        task_id = result.get("task_id")
        score = result.get("score")
        if isinstance(task_id, str) and isinstance(score, (int, float)):
            task_scores[task_id] = round(float(score), 4)

        if result.get("used_fallback") is True:
            fallback_count += 1
        if result.get("decision_parsed") is False:
            parse_failure_count += 1
        request_error_count += _request_error_count(result)

        if isinstance(result.get("steps_used"), (int, float)):
            steps.append(float(result["steps_used"]))
        reward_trace = result.get("reward_trace")
        if isinstance(reward_trace, list):
            reward_lengths.append(float(len(reward_trace)))

        report = result.get("submission_report")
        if not isinstance(report, dict):
            continue
        for source, bucket in (
            ("resolution_score", resolution_scores),
            ("evidence_score", evidence_scores),
            ("documentation_score", documentation_scores),
            ("efficiency_score", efficiency_scores),
        ):
            value = report.get(source)
            if isinstance(value, (int, float)):
                bucket.append(float(value))

    return {
        "path": str(path),
        "run_id": payload.get("run_id"),
        "model_name": payload.get("model_name"),
        "mean_score": payload.get("mean_score"),
        "raw_mean_score": payload.get("raw_mean_score"),
        "strict_baseline_scoring": payload.get("strict_baseline_scoring"),
        "task_scores": task_scores,
        "fallback_count": fallback_count,
        "parse_failure_count": parse_failure_count,
        "request_error_count": request_error_count,
        "avg_resolution_score": _safe_mean(resolution_scores),
        "avg_evidence_score": _safe_mean(evidence_scores),
        "avg_documentation_score": _safe_mean(documentation_scores),
        "avg_efficiency_score": _safe_mean(efficiency_scores),
        "avg_steps_used": _safe_mean(steps),
        "avg_reward_trace_len": _safe_mean(reward_lengths),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"path: {summary['path']}")
    print(f"run_id: {summary['run_id']}")
    print(f"model: {summary['model_name']}")
    print(
        "mean_score: "
        f"{summary['mean_score']:.4f} "
        f"(raw_mean_score={summary['raw_mean_score']:.4f}, "
        f"strict_baseline_scoring={summary['strict_baseline_scoring']})"
    )

    print("tasks:")
    for task_id in TASK_ORDER:
        score = summary["task_scores"].get(task_id)
        rendered = "-" if score is None else f"{score:.4f}"
        print(f"  {task_id}: {rendered}")

    print("components:")
    for label in (
        "avg_resolution_score",
        "avg_evidence_score",
        "avg_documentation_score",
        "avg_efficiency_score",
        "avg_steps_used",
        "avg_reward_trace_len",
    ):
        value = summary[label]
        rendered = "-" if value is None else f"{value:.4f}"
        print(f"  {label}: {rendered}")

    print("health:")
    print(f"  fallbacks: {summary['fallback_count']}")
    print(f"  parse_failures: {summary['parse_failure_count']}")
    print(f"  request_errors: {summary['request_error_count']}")


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in args.paths] if args.paths else [find_latest_eval()]
    for index, path in enumerate(paths):
        if index:
            print()
        print_summary(summarize_eval(path))


if __name__ == "__main__":
    main()
