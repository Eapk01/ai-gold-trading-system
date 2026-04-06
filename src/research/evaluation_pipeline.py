"""Evaluation helpers for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .defaults import default_threshold_list


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation output for one prediction segment."""

    summary: Dict[str, Any]
    threshold_metrics: pd.DataFrame
    calibration_bins: pd.DataFrame
    prediction_rows: pd.DataFrame


class EvaluationPipeline:
    """Compute threshold-free, thresholded, and calibration metrics."""

    def __init__(self, thresholds: Sequence[float] | None = None, calibration_bins: int = 10) -> None:
        self.thresholds = tuple(float(value) for value in (thresholds or default_threshold_list()))
        self.calibration_bins = int(calibration_bins)

    def evaluate_binary(self, target: pd.Series, prediction: pd.Series) -> Dict[str, float]:
        """Return simple binary metrics for aligned target/prediction series."""
        target_series = pd.to_numeric(target, errors="coerce")
        prediction_series = pd.to_numeric(prediction, errors="coerce")
        valid_mask = target_series.notna() & prediction_series.notna()
        if not bool(valid_mask.any()):
            return {"rows": 0.0, "accuracy": 0.0}

        y_true = target_series.loc[valid_mask].astype(int)
        y_pred = prediction_series.loc[valid_mask].astype(int)
        accuracy = float((y_true == y_pred).mean())
        positive_predictions = int((y_pred == 1).sum())
        positive_targets = int((y_true == 1).sum())
        true_positives = int(((y_true == 1) & (y_pred == 1)).sum())
        precision = true_positives / positive_predictions if positive_predictions else 0.0
        recall = true_positives / positive_targets if positive_targets else 0.0

        return {
            "rows": float(len(y_true)),
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
        }

    def evaluate_segment(
        self,
        *,
        timestamps: pd.Index,
        target: pd.Series,
        prediction: pd.Series,
        probability: pd.Series,
        confidence: pd.Series,
        fold_name: str,
        split_segment: str,
        model_name: str,
    ) -> EvaluationResult:
        """Evaluate one fold segment using raw probabilities and fixed thresholds."""
        evaluation_rows = pd.DataFrame(
            {
                "timestamp": timestamps,
                "fold_name": fold_name,
                "split_segment": split_segment,
                "model_name": model_name,
                "target": pd.to_numeric(target, errors="coerce"),
                "prediction": pd.to_numeric(prediction, errors="coerce"),
                "probability": pd.to_numeric(probability, errors="coerce"),
                "confidence": pd.to_numeric(confidence, errors="coerce"),
            },
            index=timestamps,
        )
        evaluation_rows = evaluation_rows.dropna(subset=["target", "prediction", "probability", "confidence"]).copy()
        if evaluation_rows.empty:
            return EvaluationResult(
                summary={"rows": 0},
                threshold_metrics=pd.DataFrame(columns=["threshold", "rows_kept", "coverage_rate", "accuracy", "precision", "recall", "f1"]),
                calibration_bins=pd.DataFrame(columns=["bin", "rows", "avg_probability", "positive_rate"]),
                prediction_rows=pd.DataFrame(columns=list(evaluation_rows.columns)),
            )

        evaluation_rows["target"] = evaluation_rows["target"].astype(int)
        evaluation_rows["prediction"] = evaluation_rows["prediction"].astype(int)
        evaluation_rows["correct"] = evaluation_rows["target"] == evaluation_rows["prediction"]

        y_true = evaluation_rows["target"]
        y_pred = evaluation_rows["prediction"]
        y_prob = evaluation_rows["probability"].clip(0.0, 1.0)

        summary = {
            "rows": int(len(evaluation_rows)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": self._safe_metric(lambda: roc_auc_score(y_true, y_prob)),
            "pr_auc": self._safe_metric(lambda: average_precision_score(y_true, y_prob)),
            "brier_score": self._safe_metric(lambda: brier_score_loss(y_true, y_prob)),
            "positive_rate": float(y_true.mean()),
            "average_probability": float(y_prob.mean()),
        }

        threshold_rows = [self._build_threshold_row(evaluation_rows, threshold) for threshold in self.thresholds]
        calibration_bins = self._build_calibration_bins(evaluation_rows)
        return EvaluationResult(
            summary=summary,
            threshold_metrics=pd.DataFrame(threshold_rows),
            calibration_bins=calibration_bins,
            prediction_rows=evaluation_rows.reset_index(drop=True),
        )

    def _build_threshold_row(self, evaluation_rows: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        filtered = evaluation_rows.loc[evaluation_rows["probability"] >= threshold]
        if filtered.empty:
            return {
                "threshold": float(threshold),
                "rows_kept": 0,
                "coverage_rate": 0.0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }

        y_true = filtered["target"].astype(int)
        y_pred = filtered["prediction"].astype(int)
        return {
            "threshold": float(threshold),
            "rows_kept": int(len(filtered)),
            "coverage_rate": float(len(filtered) / len(evaluation_rows)) if len(evaluation_rows) else 0.0,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    def _build_calibration_bins(self, evaluation_rows: pd.DataFrame) -> pd.DataFrame:
        if evaluation_rows.empty:
            return pd.DataFrame(columns=["bin", "rows", "avg_probability", "positive_rate"])

        frame = evaluation_rows.copy()
        probability = frame["probability"].clip(0.0, 1.0)
        bin_edges = np.linspace(0.0, 1.0, self.calibration_bins + 1)
        frame["bin"] = pd.cut(probability, bins=bin_edges, include_lowest=True, duplicates="drop")
        grouped = frame.groupby("bin", observed=False)
        rows = []
        for bin_value, group in grouped:
            if group.empty:
                continue
            rows.append(
                {
                    "bin": str(bin_value),
                    "rows": int(len(group)),
                    "avg_probability": float(group["probability"].mean()),
                    "positive_rate": float(group["target"].mean()),
                }
            )
        return pd.DataFrame(rows)

    def _safe_metric(self, metric_fn) -> float | None:
        try:
            return float(metric_fn())
        except Exception:
            return None
