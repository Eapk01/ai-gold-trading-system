"""Unified runtime predictor loading and scoring helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from src.ai_models import AIModelManager

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None


@dataclass
class RuntimePredictor(ABC):
    """Common runtime scoring contract for promoted artifacts."""

    artifact_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def required_feature_columns(self) -> list[str]:
        """Return the feature columns required for scoring."""
        raise NotImplementedError

    @property
    def min_history_rows(self) -> int:
        """Return the minimum rows required for valid scoring."""
        return 1

    @property
    def target_column(self) -> str:
        """Return the target column associated with the loaded artifact."""
        return str(self.metadata.get("target_column") or "")

    @abstractmethod
    def predict_batch(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Score a prepared feature matrix."""
        raise NotImplementedError


@dataclass
class EnsemblePredictor(RuntimePredictor):
    """Adapter exposing the legacy AIModelManager artifact behind the shared contract."""

    manager: AIModelManager | None = None

    @property
    def required_feature_columns(self) -> list[str]:
        return list(getattr(self.manager, "feature_columns", []) or [])

    @property
    def target_column(self) -> str:
        return str(getattr(self.manager, "target_column", "") or super().target_column)

    def predict_batch(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if self.manager is None:
            raise ValueError("Ensemble predictor is missing an AIModelManager instance")
        return self.manager.predict_ensemble_batch(
            feature_frame,
            feature_columns=self.required_feature_columns,
            method="voting",
        )

    @classmethod
    def from_artifact(
        cls,
        *,
        config: Dict[str, Any],
        artifact_path: str,
        manager: AIModelManager | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> "EnsemblePredictor":
        predictor_manager = manager or AIModelManager(config)
        if not predictor_manager.load_models(str(artifact_path)):
            raise ValueError(f"Failed to load ensemble artifact: {artifact_path}")
        metadata = {
            "artifact_type": "legacy_ensemble",
            "runtime_loader": "ensemble",
            "target_column": str(getattr(predictor_manager, "target_column", "") or ""),
        }
        if isinstance(payload, dict):
            metadata.update(dict(payload.get("metadata") or {}))
        return cls(
            artifact_path=str(artifact_path),
            metadata=metadata,
            manager=predictor_manager,
        )

    @classmethod
    def from_manager(
        cls,
        manager: AIModelManager,
        *,
        artifact_path: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> "EnsemblePredictor":
        return cls(
            artifact_path=str(artifact_path),
            metadata={
                "artifact_type": "legacy_ensemble",
                "runtime_loader": "ensemble",
                "target_column": str(getattr(manager, "target_column", "") or ""),
                **dict(metadata or {}),
            },
            manager=manager,
        )


if nn is not None:
    class LSTMSequenceClassifier(nn.Module):
        """Small binary LSTM classifier used for research and runtime scoring."""

        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=int(input_size),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=effective_dropout,
            )
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, sequences):
            outputs, _ = self.lstm(sequences)
            return self.head(outputs[:, -1, :]).squeeze(-1)
else:  # pragma: no cover - exercised in environments without torch
    class LSTMSequenceClassifier:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for LSTM sequence artifacts")


@dataclass
class LSTMPredictor(RuntimePredictor):
    """Batch scorer for persisted LSTM sequence artifacts."""

    payload: Dict[str, Any] = field(default_factory=dict)
    _model: Any = None

    def __post_init__(self) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to load LSTM predictor artifacts")

        self.metadata = dict(self.payload.get("metadata") or self.metadata or {})
        model_settings = dict(self.payload.get("model_settings") or {})
        self._feature_columns = list(self.payload.get("feature_columns") or [])
        self._lookback_window = max(int(self.payload.get("lookback_window") or 1), 1)
        self._fill_values = {
            str(column): float(value)
            for column, value in dict(self.payload.get("fill_values") or {}).items()
        }
        scaler_mean = self.payload.get("scaler_mean")
        scaler_scale = self.payload.get("scaler_scale")
        self._scaler_mean = np.asarray([] if scaler_mean is None else scaler_mean, dtype=np.float64)
        self._scaler_scale = np.asarray([] if scaler_scale is None else scaler_scale, dtype=np.float64)
        self._decision_threshold = float(self.payload.get("decision_threshold") or 0.5)
        self._device = torch.device("cpu")
        self._model = LSTMSequenceClassifier(
            input_size=int(model_settings.get("input_size") or len(self._feature_columns)),
            hidden_size=int(model_settings.get("hidden_size") or 32),
            num_layers=int(model_settings.get("num_layers") or 1),
            dropout=float(model_settings.get("dropout") or 0.0),
        )
        state_dict = self.payload.get("state_dict") or {}
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    @property
    def required_feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def min_history_rows(self) -> int:
        return self._lookback_window

    @property
    def target_column(self) -> str:
        return str(self.payload.get("target_column") or super().target_column)

    def predict_batch(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_feature_frame(feature_frame)
        results = pd.DataFrame(index=frame.index)
        results["is_valid"] = False
        results["prediction"] = np.nan
        results["confidence"] = 0.0
        results["probability"] = np.nan

        if frame.empty or len(frame) < self._lookback_window:
            return results

        sequences = self._build_sequences(frame.to_numpy(dtype=np.float32))
        if sequences.size == 0:
            return results

        with torch.no_grad():
            logits = self._model(torch.tensor(sequences, dtype=torch.float32, device=self._device))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

        scored_index = frame.index[self._lookback_window - 1 :]
        prediction = (probabilities >= self._decision_threshold).astype(np.float64)
        confidence = np.maximum(probabilities, 1.0 - probabilities)
        results.loc[scored_index, "is_valid"] = True
        results.loc[scored_index, "prediction"] = prediction
        results.loc[scored_index, "confidence"] = confidence
        results.loc[scored_index, "probability"] = probabilities
        return results

    def _prepare_feature_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if not self._feature_columns:
            raise ValueError("LSTM predictor is missing required feature columns")
        missing = [column for column in self._feature_columns if column not in feature_frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        frame = feature_frame.loc[:, self._feature_columns].apply(pd.to_numeric, errors="coerce")
        if self._fill_values:
            frame = frame.fillna(self._fill_values)
        frame = frame.fillna(0.0)

        if self._scaler_mean.size and self._scaler_scale.size:
            safe_scale = np.where(self._scaler_scale == 0.0, 1.0, self._scaler_scale)
            values = (frame.to_numpy(dtype=np.float64) - self._scaler_mean) / safe_scale
            frame = pd.DataFrame(values, index=frame.index, columns=frame.columns)
        return frame

    def _build_sequences(self, values: np.ndarray) -> np.ndarray:
        sequence_count = len(values) - self._lookback_window + 1
        if sequence_count <= 0:
            return np.empty((0, self._lookback_window, values.shape[1]), dtype=np.float32)
        return np.stack(
            [
                values[offset : offset + self._lookback_window]
                for offset in range(sequence_count)
            ],
            axis=0,
        ).astype(np.float32)

    @classmethod
    def from_artifact(cls, artifact_path: str, payload: Dict[str, Any]) -> "LSTMPredictor":
        return cls(
            artifact_path=str(artifact_path),
            metadata=dict(payload.get("metadata") or {}),
            payload=dict(payload),
        )


def build_lstm_artifact_payload(
    *,
    model: Any,
    feature_columns: Iterable[str],
    lookback_window: int,
    fill_values: Dict[str, float],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    decision_threshold: float,
    target_column: str,
    model_settings: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a serialized LSTM predictor payload."""
    if torch is None:
        raise ImportError("PyTorch is required to save LSTM predictor artifacts")
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    return {
        "artifact_type": "runtime_predictor",
        "runtime_loader": "lstm",
        "target_column": str(target_column or ""),
        "feature_columns": list(feature_columns),
        "lookback_window": int(lookback_window),
        "fill_values": {str(column): float(value) for column, value in dict(fill_values or {}).items()},
        "scaler_mean": np.asarray(scaler_mean, dtype=np.float64),
        "scaler_scale": np.asarray(scaler_scale, dtype=np.float64),
        "decision_threshold": float(decision_threshold),
        "model_settings": dict(model_settings or {}),
        "state_dict": cpu_state_dict,
        "metadata": {
            "artifact_type": "runtime_predictor",
            "runtime_loader": "lstm",
            "lookback_window": int(lookback_window),
            "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
            "target_column": str(target_column or ""),
            **dict(metadata or {}),
        },
    }


def load_runtime_predictor(
    config: Dict[str, Any],
    artifact_path: str,
    *,
    manager: AIModelManager | None = None,
) -> RuntimePredictor:
    """Load a promoted/runtime artifact into a unified runtime predictor."""
    payload = joblib.load(str(artifact_path))
    if isinstance(payload, dict):
        runtime_loader = str(payload.get("runtime_loader") or "").strip().lower()
        if runtime_loader == "lstm":
            return LSTMPredictor.from_artifact(str(artifact_path), payload)
        if "models" in payload and "scalers" in payload:
            return EnsemblePredictor.from_artifact(
                config=config,
                artifact_path=str(artifact_path),
                manager=manager,
                payload=payload,
            )
    raise ValueError(f"Unsupported runtime artifact format: {artifact_path}")
