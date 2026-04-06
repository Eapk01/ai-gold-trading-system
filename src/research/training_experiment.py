"""Stage 4 and Stage 5 helpers for experiment metadata and ranking summaries."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def select_best_threshold(threshold_rows: Iterable[Dict[str, Any]], *, model_name: str) -> float | None:
    """Choose the deterministic Stage 4 threshold using validation folds only."""
    candidates = [
        row
        for row in threshold_rows
        if row.get("model_name") == model_name and row.get("split_segment") == "validation"
    ]
    if not candidates:
        return None

    def sort_key(row: Dict[str, Any]) -> tuple[float, float, float]:
        f1 = float(row.get("f1") or 0.0)
        coverage = float(row.get("coverage_rate") or 0.0)
        threshold = float(row.get("threshold") or 0.0)
        return (f1, coverage, -threshold)

    best_row = max(candidates, key=sort_key)
    return float(best_row.get("threshold"))


def build_threshold_summary_rows(threshold_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract threshold summary rows from an experiment result payload."""
    return list((threshold_summary or {}).get("rows") or [])


def summarize_selected_threshold_metrics(
    threshold_rows: Iterable[Dict[str, Any]],
    *,
    selected_threshold: float | None,
    model_name: str,
    baseline_names: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """Build validation/test summaries for the selected threshold only."""
    if selected_threshold is None:
        return {
            "validation": _empty_selected_threshold_summary(),
            "test": _empty_selected_threshold_summary(),
        }

    rows = [
        row for row in threshold_rows
        if float(row.get("threshold") or 0.0) == float(selected_threshold)
    ]
    baseline_name_set = {str(name) for name in baseline_names}
    return {
        "validation": _summarize_segment(rows, selected_threshold, model_name, baseline_name_set, "validation"),
        "test": _summarize_segment(rows, selected_threshold, model_name, baseline_name_set, "test"),
    }


def _summarize_segment(
    rows: List[Dict[str, Any]],
    selected_threshold: float,
    model_name: str,
    baseline_names: set[str],
    split_segment: str,
) -> Dict[str, float]:
    segment_rows = [row for row in rows if row.get("split_segment") == split_segment]
    model_rows = [row for row in segment_rows if row.get("model_name") == model_name]
    baseline_rows = [row for row in segment_rows if str(row.get("model_name")) in baseline_names]
    if not model_rows:
        summary = _empty_selected_threshold_summary()
        summary["selected_threshold"] = float(selected_threshold)
        return summary

    model_f1_by_fold = {
        str(row.get("fold_name")): float(row.get("f1") or 0.0)
        for row in model_rows
    }
    model_coverage_values = [float(row.get("coverage_rate") or 0.0) for row in model_rows]

    best_baseline_by_fold: Dict[str, float] = {}
    for row in baseline_rows:
        fold_name = str(row.get("fold_name"))
        best_baseline_by_fold[fold_name] = max(
            best_baseline_by_fold.get(fold_name, float("-inf")),
            float(row.get("f1") or 0.0),
        )
    beat_count = sum(
        1 for fold_name, model_f1 in model_f1_by_fold.items()
        if model_f1 > best_baseline_by_fold.get(fold_name, float("-inf"))
    )
    fold_count = len(model_f1_by_fold)
    model_f1_values = list(model_f1_by_fold.values())
    best_baseline_mean_f1 = (
        sum(best_baseline_by_fold.get(fold_name, 0.0) for fold_name in model_f1_by_fold) / fold_count
        if fold_count
        else 0.0
    )
    return {
        "selected_threshold": float(selected_threshold),
        "fold_count": float(fold_count),
        "mean_f1": float(sum(model_f1_values) / fold_count) if fold_count else 0.0,
        "f1_std": float(np.asarray(model_f1_values, dtype=np.float64).std(ddof=0)) if len(model_f1_values) > 1 else 0.0,
        "mean_coverage": float(sum(model_coverage_values) / len(model_coverage_values)) if model_coverage_values else 0.0,
        "best_baseline_mean_f1": float(best_baseline_mean_f1),
        "beat_count": float(beat_count),
        "beat_rate": float(beat_count / fold_count) if fold_count else 0.0,
    }


def _empty_selected_threshold_summary() -> Dict[str, float]:
    return {
        "selected_threshold": 0.0,
        "fold_count": 0.0,
        "mean_f1": 0.0,
        "f1_std": 0.0,
        "mean_coverage": 0.0,
        "best_baseline_mean_f1": 0.0,
        "beat_count": 0.0,
        "beat_rate": 0.0,
    }
