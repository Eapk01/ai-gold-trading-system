"""Training orchestration primitives for research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import pandas as pd

from ..schemas import TrainerOutput


TrainCallable = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], Dict[str, Any]]


@dataclass
class TrainingPipeline:
    """Minimal wrapper around a fold-local training callable."""

    trainer: TrainCallable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Execute the configured trainer and normalize its response shape."""
        result = self.trainer(train_features, train_target, test_features)
        return self._normalize_output(result)

    def run_segments(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        segments: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, Any]]:
        """Train once and predict multiple named segments when supported."""
        trainer_obj = getattr(self.trainer, "__self__", None)
        if trainer_obj is not None and hasattr(trainer_obj, "fit_predict_segments"):
            outputs = trainer_obj.fit_predict_segments(train_features, train_target, segments)
            return {name: self._normalize_output(output) for name, output in outputs.items()}

        return {
            name: self.run(train_features, train_target, segment_features)
            for name, segment_features in segments.items()
        }

    @staticmethod
    def identity_trainer(
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Simple placeholder trainer that predicts the majority class."""
        cleaned_target = pd.to_numeric(train_target, errors="coerce").dropna()
        if cleaned_target.empty:
            majority_value = 0.0
        else:
            majority_value = float(cleaned_target.mode(dropna=True).iloc[0])
        prediction = pd.Series(majority_value, index=test_features.index, dtype="float64")
        return {"prediction": prediction, "selected_features": list(train_features.columns)}

    def _normalize_output(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, TrainerOutput):
            return {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "selected_features": list(result.selected_features),
                "model_artifact_path": result.model_artifact_path,
                "metadata": dict(result.metadata),
            }
        if not isinstance(result, dict):
            raise TypeError("Training pipeline trainer must return a dictionary")
        if "prediction" not in result:
            raise ValueError("Training pipeline result must include a 'prediction' entry")
        return result
