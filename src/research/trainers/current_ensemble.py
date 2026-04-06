"""Fold-local adapter for the current ensemble model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ai_models import AIModelManager
from src.research.preprocessing import ResearchPreprocessor
from src.research.schemas import CandidateArtifact, TrainerOutput
from .base import ResearchTrainer


@dataclass
class CurrentEnsembleTrainer(ResearchTrainer):
    """Train the configured ensemble family on one research fold."""

    config: Dict
    model_names: List[str] | None = None
    target_column: str | None = None
    model_params: Dict[str, Dict[str, Any]] | None = None

    def fit_predict(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
    ) -> TrainerOutput:
        return self.fit_predict_segments(
            train_features,
            train_target,
            {"segment": test_features},
        )["segment"]

    def fit_predict_segments(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        segments: dict[str, pd.DataFrame],
    ) -> dict[str, TrainerOutput]:
        manager, preprocessor, scaler, cleaned_target, prepared_segments = self._fit_manager(
            train_features,
            train_target,
            segments,
        )
        trained_names: List[str] = list(manager.models.keys())

        outputs: dict[str, TrainerOutput] = {}
        for segment_name, test_frame in prepared_segments.items():
            scaled_test = scaler.transform(test_frame)
            positive_probabilities = {}
            for model_name, model in manager.models.items():
                if hasattr(model, "predict_proba"):
                    probabilities = np.asarray(model.predict_proba(scaled_test), dtype=np.float64)
                    if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                        positive_probabilities[model_name] = probabilities[:, 1]
                    else:
                        positive_probabilities[model_name] = probabilities.reshape(-1)
                else:
                    predictions = np.asarray(model.predict(scaled_test), dtype=np.float64)
                    positive_probabilities[model_name] = predictions

            probability_frame = pd.DataFrame(positive_probabilities, index=test_frame.index)
            positive_probability = probability_frame.mean(axis=1).clip(0.0, 1.0)
            prediction = (positive_probability >= 0.5).astype(float)
            confidence = positive_probability.where(positive_probability >= 0.5, 1.0 - positive_probability)

            outputs[segment_name] = TrainerOutput(
                prediction=prediction.astype("float64"),
                confidence=confidence.astype("float64"),
                probabilities=positive_probability.astype("float64"),
                selected_features=list(train_features.columns),
                metadata={
                    "trainer_name": "current_ensemble",
                    "trained_model_names": trained_names,
                    "model_count": len(trained_names),
                    "preprocessing": "median_imputation_then_standard_scaling",
                    "model_params": dict(self.model_params or {}),
                },
            )
        return outputs

    def fit_candidate_artifact(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        artifact_path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateArtifact:
        manager, preprocessor, _, cleaned_target, _ = self._fit_manager(train_features, train_target, {})
        manager.training_history.append(
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "target_column": manager.target_column,
                "feature_count": len(train_features.columns),
                "training_samples": int(len(cleaned_target)),
                "models_trained": list(manager.models.keys()),
                "research_candidate": True,
                "metadata": dict(metadata or {}),
            }
        )
        manager.save_models(artifact_path)
        return CandidateArtifact(
            artifact_path=str(artifact_path),
            selected_features=list(train_features.columns),
            trainer_name="current_ensemble",
            metadata={
                "trained_model_names": list(manager.models.keys()),
                "model_count": len(manager.models),
                "preprocessing": "median_imputation_then_standard_scaling",
                "fill_value_columns": sorted(preprocessor.fill_values.keys()),
                "model_params": dict(self.model_params or {}),
                **dict(metadata or {}),
            },
        )

    def _fit_manager(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        segments: dict[str, pd.DataFrame],
    ) -> tuple[AIModelManager, ResearchPreprocessor, StandardScaler, pd.Series, dict[str, pd.DataFrame]]:
        manager = AIModelManager(self.config)
        manager.feature_columns = list(train_features.columns)
        if self.target_column:
            manager.target_column = self.target_column

        candidate_model_names = list(self.model_names or manager.model_names)
        train_frame = train_features.apply(pd.to_numeric, errors="coerce")
        cleaned_target = pd.to_numeric(train_target, errors="coerce")
        normalized_segments = {
            name: segment.apply(pd.to_numeric, errors="coerce")
            for name, segment in segments.items()
        }
        valid_train_mask = cleaned_target.notna()
        train_frame = train_frame.loc[valid_train_mask]
        cleaned_target = cleaned_target.loc[valid_train_mask]
        if train_frame.empty or cleaned_target.empty:
            raise ValueError("Current ensemble trainer requires at least one non-null training target row")

        preprocessor = ResearchPreprocessor().fit(train_frame)
        prepared_train = preprocessor.transform(train_frame)
        prepared_segments = {
            name: preprocessor.transform(segment_frame)
            for name, segment_frame in normalized_segments.items()
        }

        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(prepared_train)
        manager.scalers[manager.target_column] = scaler
        manager.preprocessors[manager.target_column] = dict(preprocessor.fill_values)

        trained_models = {}
        trained_names: List[str] = []
        configured_model_params = self._research_safe_model_params()
        for model_name in candidate_model_names:
            model = manager._get_model_instance(
                model_name,
                task_type="classification",
                model_params=configured_model_params.get(model_name, {}),
            )
            if model is None:
                continue
            model.fit(scaled_train, cleaned_target)
            trained_models[model_name] = model
            trained_names.append(model_name)

        if not trained_models:
            raise ValueError("No ensemble models could be trained for the current fold")

        manager.models = trained_models
        manager.model_performance = {
            model_name: {
                "model_name": model_name,
                "research_candidate": True,
                "model_params": configured_model_params.get(model_name, {}),
            }
            for model_name in trained_names
        }
        return manager, preprocessor, scaler, cleaned_target, prepared_segments

    def _research_safe_model_params(self) -> Dict[str, Dict[str, Any]]:
        configured_model_params = {
            str(model_name): dict(params or {})
            for model_name, params in dict(self.model_params or {}).items()
        }
        for model_name in list(self.model_names or []):
            configured_model_params.setdefault(str(model_name), {})

        for model_name, params in configured_model_params.items():
            normalized_name = str(model_name).strip().lower()
            if normalized_name == "random_forest":
                params.setdefault("n_jobs", 1)
            elif normalized_name == "xgboost":
                params.setdefault("n_jobs", 1)
                params.setdefault("nthread", 1)
            elif normalized_name == "lightgbm":
                params.setdefault("n_jobs", 1)
        return configured_model_params
