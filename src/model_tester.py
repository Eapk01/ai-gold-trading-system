"""
Model evaluation helpers for prediction-quality testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_CONFIDENCE_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("0.50-0.55", 0.50, 0.55),
    ("0.55-0.60", 0.55, 0.60),
    ("0.60-0.65", 0.60, 0.65),
    ("0.65-0.70", 0.65, 0.70),
    ("0.70+", 0.70, None),
)


@dataclass
class ModelTestResult:
    summary: Dict[str, Any]
    threshold_performance: pd.DataFrame
    confidence_buckets: pd.DataFrame
    row_evaluations: pd.DataFrame


class ModelTester:
    """Evaluate prediction quality across all scorable rows."""

    def __init__(
        self,
        *,
        confidence_thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
        confidence_buckets: Sequence[tuple[str, float, float | None]] = DEFAULT_CONFIDENCE_BUCKETS,
    ) -> None:
        self.confidence_thresholds = tuple(float(value) for value in confidence_thresholds)
        self.confidence_buckets = tuple(confidence_buckets)

    def evaluate(
        self,
        prepared_data: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        target_series: pd.Series,
    ) -> ModelTestResult:
        total_rows = int(len(prepared_data))
        valid_prediction_rows = int(prediction_frame.get("is_valid", pd.Series(dtype=bool)).fillna(False).sum())

        aligned_target = pd.to_numeric(target_series.reindex(prediction_frame.index), errors="coerce")
        valid_mask = prediction_frame["is_valid"].fillna(False).astype(bool)
        target_present_mask = aligned_target.notna()
        scored_mask = valid_mask & target_present_mask

        row_evaluations = pd.DataFrame(index=prediction_frame.index[scored_mask])
        if scored_mask.any():
            row_evaluations["target"] = aligned_target.loc[scored_mask].astype(int)
            row_evaluations["prediction"] = prediction_frame.loc[scored_mask, "prediction"].astype(int)
            row_evaluations["confidence"] = prediction_frame.loc[scored_mask, "confidence"].astype(float)
            row_evaluations["correct"] = (
                row_evaluations["target"].astype(int) == row_evaluations["prediction"].astype(int)
            )
            row_evaluations = row_evaluations.reset_index().rename(columns={row_evaluations.index.name or "index": "timestamp"})
        else:
            row_evaluations = pd.DataFrame(columns=["timestamp", "target", "prediction", "confidence", "correct"])

        invalid_rows = max(total_rows - valid_prediction_rows, 0)
        target_missing_rows = int((valid_mask & ~target_present_mask).sum())
        scored_rows = int(len(row_evaluations))

        summary = {
            "total_rows": total_rows,
            "valid_prediction_rows": valid_prediction_rows,
            "invalid_rows": invalid_rows,
            "target_missing_rows": target_missing_rows,
            "scored_rows": scored_rows,
            "coverage_rate": (scored_rows / total_rows) if total_rows else 0.0,
        }
        summary.update(self._compute_metrics(row_evaluations))

        threshold_rows = [
            self._build_threshold_row(row_evaluations, threshold, total_rows)
            for threshold in self.confidence_thresholds
        ]
        threshold_performance = pd.DataFrame(threshold_rows)

        bucket_rows = [
            self._build_bucket_row(row_evaluations, label, lower, upper, total_rows)
            for label, lower, upper in self.confidence_buckets
        ]
        confidence_buckets = pd.DataFrame(bucket_rows)

        return ModelTestResult(
            summary=summary,
            threshold_performance=threshold_performance,
            confidence_buckets=confidence_buckets,
            row_evaluations=row_evaluations,
        )

    def _build_threshold_row(self, row_evaluations: pd.DataFrame, threshold: float, total_rows: int) -> Dict[str, Any]:
        filtered = row_evaluations.loc[row_evaluations["confidence"] >= threshold]
        row = {
            "threshold": threshold,
            "rows_kept": int(len(filtered)),
            "coverage_rate": (len(filtered) / total_rows) if total_rows else 0.0,
        }
        row.update(self._compute_metrics(filtered))
        return row

    def _build_bucket_row(
        self,
        row_evaluations: pd.DataFrame,
        label: str,
        lower: float,
        upper: float | None,
        total_rows: int,
    ) -> Dict[str, Any]:
        mask = row_evaluations["confidence"] >= lower
        if upper is not None:
            mask &= row_evaluations["confidence"] < upper
        filtered = row_evaluations.loc[mask]
        row = {
            "bucket": label,
            "rows": int(len(filtered)),
            "coverage_rate": (len(filtered) / total_rows) if total_rows else 0.0,
        }
        row.update(self._compute_metrics(filtered))
        return row

    def _compute_metrics(self, row_evaluations: pd.DataFrame) -> Dict[str, Any]:
        if row_evaluations.empty:
            return {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "positive_precision": None,
                "positive_recall": None,
                "positive_f1": None,
                "negative_precision": None,
                "negative_recall": None,
                "negative_f1": None,
                "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            }

        y_true = row_evaluations["target"].astype(int)
        y_pred = row_evaluations["prediction"].astype(int)

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "positive_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "positive_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "negative_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "negative_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "negative_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
