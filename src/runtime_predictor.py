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
        """Versioned binary LSTM classifier used for research and runtime scoring."""

        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            dense_hidden_size: int,
            dense_dropout: float,
            activation: str,
            bidirectional: bool,
        ) -> None:
            super().__init__()
            effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
            self.bidirectional = bool(bidirectional)
            self.lstm = nn.LSTM(
                input_size=int(input_size),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=effective_dropout,
                bidirectional=self.bidirectional,
            )
            output_size = int(hidden_size) * (2 if self.bidirectional else 1)
            dense_size = max(int(dense_hidden_size), 1)
            self.norm = nn.LayerNorm(output_size)
            activation_layer = nn.GELU() if str(activation).strip().lower() == "gelu" else nn.ReLU()
            self.head = nn.Sequential(
                nn.Linear(output_size, dense_size),
                activation_layer,
                nn.Dropout(float(dense_dropout)),
                nn.Linear(dense_size, 1),
            )

        def forward(self, sequences):
            _outputs, (hidden_state, _cell_state) = self.lstm(sequences)
            if self.bidirectional:
                final_state = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
            else:
                final_state = hidden_state[-1]
            return self.head(self.norm(final_state)).squeeze(-1)
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
        artifact_version = int(self.payload.get("artifact_version") or self.metadata.get("artifact_version") or 0)
        if artifact_version != 2:
            raise ValueError("Unsupported LSTM artifact version. Retrain the LSTM candidate with artifact_version=2.")
        model_settings = dict(self.payload.get("model_settings") or {})
        self._feature_columns = list(self.payload.get("feature_columns") or [])
        self._sequence_feature_columns = list(self.payload.get("sequence_feature_columns") or self._feature_columns)
        self._engineered_feature_columns = list(self.payload.get("engineered_feature_columns") or [])
        self._feature_mode = str(self.payload.get("feature_mode") or self.metadata.get("feature_mode") or "engineered")
        self._raw_column_mapping = {
            str(key): str(value)
            for key, value in dict(self.payload.get("raw_column_mapping") or {}).items()
        }
        self._lookback_window = max(int(self.payload.get("lookback_window") or 1), 1)
        self._fill_values = {
            str(column): float(value)
            for column, value in dict(self.payload.get("fill_values") or {}).items()
        }
        scaler_mean = self.payload.get("scaler_mean")
        scaler_scale = self.payload.get("scaler_scale")
        self._scaler_mean = np.asarray([] if scaler_mean is None else scaler_mean, dtype=np.float64)
        self._scaler_scale = np.asarray([] if scaler_scale is None else scaler_scale, dtype=np.float64)
        selected_threshold = _optional_float(
            self.payload.get("selected_threshold"),
            self.metadata.get("selected_threshold"),
        )
        decision_threshold = _optional_float(
            self.payload.get("decision_threshold"),
            self.metadata.get("decision_threshold"),
            0.5,
        )
        self._decision_threshold = float(selected_threshold if selected_threshold is not None else decision_threshold)
        self._threshold_source = str(
            self.payload.get("threshold_source")
            or self.metadata.get("threshold_source")
            or ("validation_selected" if selected_threshold is not None else "trainer_default")
        )
        self.metadata = {
            **self.metadata,
            "selected_threshold": selected_threshold,
            "decision_threshold": float(decision_threshold),
            "threshold_source": self._threshold_source,
            "effective_decision_threshold": self._decision_threshold,
        }
        self._device = torch.device("cpu")
        self._model = LSTMSequenceClassifier(
            input_size=int(model_settings.get("input_size") or len(self._sequence_feature_columns)),
            hidden_size=int(model_settings.get("hidden_size") or 32),
            num_layers=int(model_settings.get("num_layers") or 1),
            dropout=float(model_settings.get("dropout") or 0.0),
            dense_hidden_size=int(model_settings.get("dense_hidden_size") or model_settings.get("hidden_size") or 32),
            dense_dropout=float(model_settings.get("dense_dropout") or 0.0),
            activation=str(model_settings.get("activation") or "gelu"),
            bidirectional=bool(model_settings.get("bidirectional", False)),
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
    def decision_threshold(self) -> float:
        return float(self._decision_threshold)

    @property
    def threshold_source(self) -> str:
        return str(self._threshold_source)

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

        frame = _build_lstm_sequence_feature_frame(
            feature_frame,
            feature_mode=self._feature_mode,
            engineered_feature_columns=self._engineered_feature_columns or self._feature_columns,
            raw_column_mapping=self._raw_column_mapping,
        )
        frame = frame.loc[:, self._sequence_feature_columns].apply(pd.to_numeric, errors="coerce")
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
    feature_mode: str,
    engineered_feature_columns: Iterable[str],
    raw_column_mapping: Dict[str, str],
    sequence_feature_columns: Iterable[str],
    selected_threshold: float | None = None,
    threshold_source: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a serialized LSTM predictor payload."""
    if torch is None:
        raise ImportError("PyTorch is required to save LSTM predictor artifacts")
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    metadata_payload = dict(metadata or {})
    raw_decision_threshold = _optional_float(decision_threshold, 0.5)
    resolved_decision_threshold = float(raw_decision_threshold if raw_decision_threshold is not None else 0.5)
    resolved_selected_threshold = _optional_float(
        selected_threshold,
        metadata_payload.get("selected_threshold"),
    )
    resolved_threshold_source = str(
        threshold_source
        or metadata_payload.get("threshold_source")
        or ("validation_selected" if resolved_selected_threshold is not None else "trainer_default")
    )
    feature_column_list = list(feature_columns)
    sequence_feature_column_list = list(sequence_feature_columns)
    engineered_feature_column_list = list(engineered_feature_columns)
    architecture_name = str((model_settings or {}).get("architecture_name") or "lstm_v2_dense_head")
    return {
        "artifact_version": 2,
        "artifact_type": "runtime_predictor",
        "runtime_loader": "lstm",
        "architecture_name": architecture_name,
        "feature_mode": str(feature_mode or "engineered"),
        "target_column": str(target_column or ""),
        "feature_columns": feature_column_list,
        "engineered_feature_columns": engineered_feature_column_list,
        "raw_column_mapping": dict(raw_column_mapping or {}),
        "sequence_feature_columns": sequence_feature_column_list,
        "lookback_window": int(lookback_window),
        "fill_values": {str(column): float(value) for column, value in dict(fill_values or {}).items()},
        "scaler_mean": np.asarray(scaler_mean, dtype=np.float64),
        "scaler_scale": np.asarray(scaler_scale, dtype=np.float64),
        "decision_threshold": resolved_decision_threshold,
        "selected_threshold": resolved_selected_threshold,
        "threshold_source": resolved_threshold_source,
        "model_settings": dict(model_settings or {}),
        "state_dict": cpu_state_dict,
        "metadata": {
            **metadata_payload,
            "artifact_version": 2,
            "artifact_type": "runtime_predictor",
            "runtime_loader": "lstm",
            "architecture_name": architecture_name,
            "feature_mode": str(feature_mode or "engineered"),
            "lookback_window": int(lookback_window),
            "feature_count": len(feature_column_list),
            "sequence_feature_count": len(sequence_feature_column_list),
            "sequence_feature_columns": sequence_feature_column_list,
            "engineered_feature_columns": engineered_feature_column_list,
            "raw_column_mapping": dict(raw_column_mapping or {}),
            "decision_threshold": resolved_decision_threshold,
            "selected_threshold": resolved_selected_threshold,
            "threshold_source": resolved_threshold_source,
            "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
            "target_column": str(target_column or ""),
        },
    }


def _optional_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(converted):
            return converted
    return None


def _build_lstm_sequence_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    feature_mode: str,
    engineered_feature_columns: Iterable[str],
    raw_column_mapping: Dict[str, str],
) -> pd.DataFrame:
    normalized_mode = str(feature_mode or "engineered").strip().lower()
    frames = []
    if normalized_mode in {"engineered", "combined"}:
        engineered_columns = [column for column in engineered_feature_columns if column in feature_frame.columns]
        if engineered_columns:
            frames.append(feature_frame.loc[:, engineered_columns].apply(pd.to_numeric, errors="coerce"))
    if normalized_mode in {"raw_market", "combined"}:
        frames.append(_build_raw_market_channels(feature_frame, raw_column_mapping=raw_column_mapping))
    if not frames:
        raise ValueError(f"LSTM v2 feature mode produced no sequence channels: {feature_mode}")
    return pd.concat(frames, axis=1)


def _build_raw_market_channels(feature_frame: pd.DataFrame, *, raw_column_mapping: Dict[str, str]) -> pd.DataFrame:
    missing = [name for name in ("open", "high", "low", "close", "volume") if name not in raw_column_mapping]
    if missing:
        raise ValueError(f"LSTM v2 raw_market mode is missing raw column mappings: {missing}")

    open_price = pd.to_numeric(feature_frame[raw_column_mapping["open"]], errors="coerce")
    high = pd.to_numeric(feature_frame[raw_column_mapping["high"]], errors="coerce")
    low = pd.to_numeric(feature_frame[raw_column_mapping["low"]], errors="coerce")
    close = pd.to_numeric(feature_frame[raw_column_mapping["close"]], errors="coerce")
    volume = pd.to_numeric(feature_frame[raw_column_mapping["volume"]], errors="coerce")
    price_range = (high - low).replace(0.0, np.nan)
    candle_max = pd.concat([open_price, close], axis=1).max(axis=1)
    candle_min = pd.concat([open_price, close], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "raw_open": open_price,
            "raw_high": high,
            "raw_low": low,
            "raw_close": close,
            "raw_volume": volume,
            "raw_return": close.pct_change(),
            "raw_body": close - open_price,
            "raw_range": high - low,
            "raw_upper_wick": high - candle_max,
            "raw_lower_wick": candle_min - low,
            "raw_close_location": ((close - low) / price_range).fillna(0.5),
        },
        index=feature_frame.index,
    )


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
