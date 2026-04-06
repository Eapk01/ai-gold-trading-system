"""Baseline predictors for research comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .labels import build_observed_persistence_labels


@dataclass
class BaselinePrediction:
    """Structured baseline output."""

    prediction: pd.Series
    confidence: pd.Series
    probability: pd.Series


class MajorityClassBaseline:
    """Baseline that predicts the train-fold majority class."""

    def predict(self, train_target: pd.Series, target_index: pd.Index) -> BaselinePrediction:
        cleaned_target = pd.to_numeric(train_target, errors="coerce").dropna()
        if cleaned_target.empty:
            positive_rate = 0.5
        else:
            positive_rate = float(cleaned_target.mean())

        predicted_value = 1.0 if positive_rate >= 0.5 else 0.0
        probability = pd.Series(positive_rate, index=target_index, dtype="float64")
        prediction = pd.Series(predicted_value, index=target_index, dtype="float64")
        confidence = probability.where(probability >= 0.5, 1.0 - probability)
        return BaselinePrediction(prediction=prediction, confidence=confidence.astype("float64"), probability=probability)


class PersistenceBaseline:
    """Baseline that repeats the previous observed target value."""

    def predict(
        self,
        feature_frame: pd.DataFrame,
        target_index: pd.Index,
        *,
        target_spec: Any,
        fallback_target: pd.Series | None = None,
    ) -> BaselinePrediction:
        required_price_column = getattr(target_spec, "price_column", None) if target_spec is not None else None
        if target_spec is not None and required_price_column in feature_frame.columns:
            prediction = pd.to_numeric(
                build_observed_persistence_labels(feature_frame, target_spec).reindex(target_index),
                errors="coerce",
            )
        elif fallback_target is not None:
            cleaned_target = pd.to_numeric(fallback_target, errors="coerce")
            prediction = cleaned_target.shift(1).reindex(target_index)
        else:
            prediction = pd.Series(index=target_index, dtype="float64")
        probability = prediction.fillna(0.5).astype("float64")
        confidence = probability.where(probability >= 0.5, 1.0 - probability)
        return BaselinePrediction(
            prediction=prediction.astype("float64"),
            confidence=confidence.astype("float64"),
            probability=probability,
        )
