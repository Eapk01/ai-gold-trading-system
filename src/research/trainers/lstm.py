"""PyTorch LSTM trainer for sequence-based research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.research.preprocessing import ResearchPreprocessor
from src.research.schemas import CandidateArtifact, TrainerOutput
from src.runtime_predictor import (
    LSTMSequenceClassifier,
    _build_lstm_sequence_feature_frame,
    build_lstm_artifact_payload,
    torch,
)
from .base import ResearchTrainer

if torch is not None:
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - exercised in environments without torch
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class LSTMTrainer(ResearchTrainer):
    """Train the replacement LSTM v2 classifier on fold-local rolling windows."""

    config: Dict
    target_column: str | None = None
    model_params: Dict[str, Any] | None = None

    @staticmethod
    def cuda_is_available() -> bool:
        """Return whether PyTorch can currently train on CUDA."""
        return bool(torch is not None and torch.cuda.is_available())

    @staticmethod
    def cuda_device_name() -> str:
        """Return the active CUDA device name when available."""
        if not LSTMTrainer.cuda_is_available():
            return ""
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return ""

    @staticmethod
    def cuda_probe_error() -> str:
        """Return an error message when CUDA is visible but cannot execute kernels."""
        if not LSTMTrainer.cuda_is_available():
            return "CUDA is not available to PyTorch"
        try:
            probe = (torch.ones(1, device="cuda") + 1.0).sum()
            float(probe.detach().cpu().item())
            torch.cuda.synchronize()
            return ""
        except Exception as exc:
            return str(exc)

    @staticmethod
    def resolve_training_device(requested_device: str = "auto") -> Dict[str, Any]:
        """Resolve the LSTM training device policy without moving any tensors."""
        if torch is None:
            raise ImportError("PyTorch is required to use the research LSTM trainer")
        requested = str(requested_device or "auto").strip().lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("Unsupported LSTM device. Use one of: auto, cpu, cuda")

        cuda_available = LSTMTrainer.cuda_is_available()
        cuda_probe_error = LSTMTrainer.cuda_probe_error() if cuda_available else ""
        cuda_usable = cuda_available and not cuda_probe_error
        if requested == "cuda" and not cuda_available:
            raise ValueError("LSTM device='cuda' was requested, but CUDA is not available. Install a CUDA-enabled PyTorch build or use device='auto'/'cpu'.")
        if requested == "cuda" and not cuda_usable:
            raise ValueError(
                "LSTM device='cuda' was requested, but CUDA cannot execute PyTorch kernels on this GPU. "
                f"PyTorch reported: {cuda_probe_error}"
            )

        training_device = "cuda" if requested in {"auto", "cuda"} and cuda_usable else "cpu"
        return {
            "requested_device": requested,
            "training_device": training_device,
            "cuda_available": cuda_available,
            "cuda_usable": cuda_usable,
            "cuda_probe_error": cuda_probe_error,
            "cuda_device_name": LSTMTrainer.cuda_device_name() if cuda_available else "",
            "torch_version": str(getattr(torch, "__version__", "")),
        }

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
        model, prepared = self._fit_model(train_features, train_target)
        lookback_window = prepared["lookback_window"]
        outputs: dict[str, TrainerOutput] = {}
        context_frame = prepared["scaled_train_frame"]

        for segment_name, raw_segment in segments.items():
            sequence_segment = self._build_sequence_frame(
                raw_segment,
                feature_mode=prepared["feature_mode"],
                engineered_feature_columns=prepared["engineered_feature_columns"],
                raw_column_mapping=prepared["raw_column_mapping"],
            )
            prepared_segment = prepared["preprocessor"].transform(sequence_segment)
            scaled_segment_values = prepared["scaler"].transform(prepared_segment)
            scaled_segment = pd.DataFrame(
                scaled_segment_values,
                index=prepared_segment.index,
                columns=prepared_segment.columns,
            )

            prediction = pd.Series(np.nan, index=raw_segment.index, dtype="float64")
            confidence = pd.Series(np.nan, index=raw_segment.index, dtype="float64")
            probabilities = pd.Series(np.nan, index=raw_segment.index, dtype="float64")

            segment_inputs = pd.concat([context_frame.tail(max(lookback_window - 1, 0)), scaled_segment], axis=0)
            if len(segment_inputs) >= lookback_window and not scaled_segment.empty:
                sequence_values, scored_index = self._build_segment_sequences(
                    context_frame=context_frame,
                    segment_frame=scaled_segment,
                    lookback_window=lookback_window,
                )
                if len(scored_index) > 0:
                    probability_values = self._predict_probabilities(model, sequence_values)
                    prediction.loc[scored_index] = (probability_values >= prepared["decision_threshold"]).astype(np.float64)
                    probabilities.loc[scored_index] = probability_values
                    confidence.loc[scored_index] = np.maximum(probability_values, 1.0 - probability_values)

            outputs[segment_name] = TrainerOutput(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                selected_features=list(train_features.columns),
                metadata={
                    "trainer_name": "lstm",
                    "lookback_window": int(lookback_window),
                    "architecture_name": prepared["architecture_name"],
                    "feature_mode": prepared["feature_mode"],
                    "sequence_feature_count": len(prepared["sequence_feature_columns"]),
                    "dense_head_summary": prepared["dense_head_summary"],
                    "bidirectional": prepared["model_settings"]["bidirectional"],
                    "training_device": prepared["device_info"]["training_device"],
                    "cuda_available": prepared["device_info"]["cuda_available"],
                    "cuda_usable": prepared["device_info"]["cuda_usable"],
                    "cuda_probe_error": prepared["device_info"]["cuda_probe_error"],
                    "cuda_device_name": prepared["device_info"]["cuda_device_name"],
                    "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
                    "model_params": dict(prepared["model_settings"]),
                },
            )
            context_frame = pd.concat([context_frame, scaled_segment], axis=0)

        return outputs

    def fit_candidate_artifact(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        artifact_path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateArtifact:
        model, prepared = self._fit_model(train_features, train_target)
        metadata_payload = dict(metadata or {})
        selected_threshold = self._optional_float(metadata_payload.get("selected_threshold"))
        decision_threshold = float(prepared["decision_threshold"])
        threshold_source = str(
            metadata_payload.get("threshold_source")
            or ("validation_selected" if selected_threshold is not None else "trainer_default")
        )
        artifact_payload = build_lstm_artifact_payload(
            model=model,
            feature_columns=train_features.columns,
            lookback_window=prepared["lookback_window"],
            fill_values=prepared["preprocessor"].fill_values,
            scaler_mean=np.asarray(prepared["scaler"].mean_, dtype=np.float64),
            scaler_scale=np.asarray(prepared["scaler"].scale_, dtype=np.float64),
            decision_threshold=decision_threshold,
            selected_threshold=selected_threshold,
            threshold_source=threshold_source,
            target_column=str(self.target_column or ""),
            model_settings=dict(prepared["model_settings"]),
            feature_mode=prepared["feature_mode"],
            engineered_feature_columns=prepared["engineered_feature_columns"],
            raw_column_mapping=prepared["raw_column_mapping"],
            sequence_feature_columns=prepared["sequence_feature_columns"],
            metadata={
                **metadata_payload,
                "trainer_name": "lstm",
                "artifact_version": 2,
                "architecture_name": prepared["architecture_name"],
                "feature_mode": prepared["feature_mode"],
                "selected_features": list(train_features.columns),
                "feature_count": len(train_features.columns),
                "sequence_feature_count": len(prepared["sequence_feature_columns"]),
                "sequence_feature_columns": list(prepared["sequence_feature_columns"]),
                "engineered_feature_columns": list(prepared["engineered_feature_columns"]),
                "raw_column_mapping": dict(prepared["raw_column_mapping"]),
                "dense_head_summary": prepared["dense_head_summary"],
                "bidirectional": prepared["model_settings"]["bidirectional"],
                "training_device": prepared["device_info"]["training_device"],
                "cuda_available": prepared["device_info"]["cuda_available"],
                "cuda_usable": prepared["device_info"]["cuda_usable"],
                "cuda_probe_error": prepared["device_info"]["cuda_probe_error"],
                "cuda_device_name": prepared["device_info"]["cuda_device_name"],
                "decision_threshold": decision_threshold,
                "selected_threshold": selected_threshold,
                "threshold_source": threshold_source,
                "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
                "trainer_params": dict(metadata_payload.get("trainer_params") or self.model_params or {}),
            },
        )
        joblib.dump(artifact_payload, artifact_path)
        return CandidateArtifact(
            artifact_path=str(artifact_path),
            selected_features=list(train_features.columns),
            trainer_name="lstm",
            metadata={
                **metadata_payload,
                "artifact_version": 2,
                "artifact_type": "runtime_predictor",
                "runtime_loader": "lstm",
                "architecture_name": prepared["architecture_name"],
                "feature_mode": prepared["feature_mode"],
                "lookback_window": int(prepared["lookback_window"]),
                "feature_count": len(train_features.columns),
                "sequence_feature_count": len(prepared["sequence_feature_columns"]),
                "sequence_feature_columns": list(prepared["sequence_feature_columns"]),
                "engineered_feature_columns": list(prepared["engineered_feature_columns"]),
                "raw_column_mapping": dict(prepared["raw_column_mapping"]),
                "dense_head_summary": prepared["dense_head_summary"],
                "bidirectional": prepared["model_settings"]["bidirectional"],
                "training_device": prepared["device_info"]["training_device"],
                "cuda_available": prepared["device_info"]["cuda_available"],
                "cuda_usable": prepared["device_info"]["cuda_usable"],
                "cuda_probe_error": prepared["device_info"]["cuda_probe_error"],
                "cuda_device_name": prepared["device_info"]["cuda_device_name"],
                "decision_threshold": decision_threshold,
                "selected_threshold": selected_threshold,
                "threshold_source": threshold_source,
                "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
                "model_params": dict(prepared["model_settings"]),
                "target_column": str(self.target_column or ""),
                "trainer_params": dict(metadata_payload.get("trainer_params") or self.model_params or {}),
            },
        )

    def _fit_model(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
    ) -> tuple[Any, Dict[str, Any]]:
        if torch is None:
            raise ImportError("PyTorch is required to use the research LSTM trainer")

        config = self._resolved_config()
        engineered_feature_columns = self._resolve_engineered_feature_columns(train_features, config["feature_mode"])
        raw_column_mapping = self._resolve_raw_column_mapping(train_features, config["feature_mode"])
        train_frame = self._build_sequence_frame(
            train_features,
            feature_mode=config["feature_mode"],
            engineered_feature_columns=engineered_feature_columns,
            raw_column_mapping=raw_column_mapping,
        )
        cleaned_target = pd.to_numeric(train_target, errors="coerce")
        valid_mask = cleaned_target.notna()
        train_frame = train_frame.loc[valid_mask]
        cleaned_target = cleaned_target.loc[valid_mask]

        if train_frame.empty or cleaned_target.empty:
            raise ValueError("LSTM trainer requires at least one non-null training target row")

        if len(train_frame) < config["lookback_window"]:
            raise ValueError(
                f"LSTM trainer requires at least {config['lookback_window']} train rows for one sequence window"
            )

        preprocessor = ResearchPreprocessor().fit(train_frame)
        prepared_train = preprocessor.transform(train_frame)
        scaler = StandardScaler()
        scaled_train_values = scaler.fit_transform(prepared_train)
        scaled_train_frame = pd.DataFrame(
            scaled_train_values,
            index=prepared_train.index,
            columns=prepared_train.columns,
        )
        train_sequences, train_labels = self._build_train_sequences(
            scaled_train_frame.to_numpy(dtype=np.float32),
            cleaned_target.to_numpy(dtype=np.float32),
            config["lookback_window"],
        )
        if len(train_sequences) == 0:
            raise ValueError("LSTM trainer could not build any train sequences from the provided fold")

        model = self._train_network(train_sequences, train_labels, config)
        return model, {
            "preprocessor": preprocessor,
            "scaler": scaler,
            "scaled_train_frame": scaled_train_frame,
            "lookback_window": config["lookback_window"],
            "decision_threshold": config["decision_threshold"],
            "feature_mode": config["feature_mode"],
            "architecture_name": config["architecture_name"],
            "sequence_feature_columns": list(scaled_train_frame.columns),
            "engineered_feature_columns": list(engineered_feature_columns),
            "raw_column_mapping": dict(raw_column_mapping),
            "dense_head_summary": self._dense_head_summary(config),
            "device_info": dict(config["device_info"]),
            "model_settings": {
                "input_size": int(train_sequences.shape[2]),
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                "dropout": config["dropout"],
                "dense_hidden_size": config["dense_hidden_size"],
                "dense_dropout": config["dense_dropout"],
                "activation": config["activation"],
                "bidirectional": config["bidirectional"],
                "architecture_name": config["architecture_name"],
            },
        }

    def _train_network(self, sequences: np.ndarray, labels: np.ndarray, config: Dict[str, Any]):
        torch.manual_seed(int(config["seed"]))
        device = torch.device(str(config["device_info"]["training_device"]))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(config["seed"]))
        model = LSTMSequenceClassifier(
            input_size=int(sequences.shape[2]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
            dense_hidden_size=int(config["dense_hidden_size"]),
            dense_dropout=float(config["dense_dropout"]),
            activation=str(config["activation"]),
            bidirectional=bool(config["bidirectional"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        criterion = self._build_loss(labels, config, device=device)

        train_sequences, train_labels, eval_sequences, eval_labels = self._split_train_eval(
            sequences,
            labels,
            lookback_window=int(config["lookback_window"]),
        )
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(train_sequences, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32),
            ),
            batch_size=max(1, int(config["batch_size"])),
            shuffle=False,
        )
        eval_tensor = (
            torch.tensor(eval_sequences, dtype=torch.float32, device=device),
            torch.tensor(eval_labels, dtype=torch.float32, device=device),
        ) if len(eval_sequences) else None

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_loss = float("inf")
        patience = max(1, int(config["early_stopping_patience"]))
        patience_left = patience

        for _epoch in range(max(1, int(config["epochs"]))):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            if eval_tensor is None:
                current_loss = float(loss.detach().cpu().item())
            else:
                with torch.no_grad():
                    eval_logits = model(eval_tensor[0])
                    current_loss = float(criterion(eval_logits, eval_tensor[1]).detach().cpu().item())

            if current_loss + 1e-6 < best_loss:
                best_loss = current_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        return model

    def _build_loss(self, labels: np.ndarray, config: Dict[str, Any], *, device: Any):
        if not bool(config["class_weighting"]):
            return nn.BCEWithLogitsLoss()

        positive_count = float(np.sum(labels == 1.0))
        negative_count = float(np.sum(labels == 0.0))
        if positive_count <= 0.0 or negative_count <= 0.0:
            return nn.BCEWithLogitsLoss()

        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _split_train_eval(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        *,
        lookback_window: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(sequences) < max(5, lookback_window * 2):
            return sequences, labels, np.empty((0,) + sequences.shape[1:], dtype=np.float32), np.empty((0,), dtype=np.float32)

        eval_size = max(1, min(len(sequences) // 5, len(sequences) - 1))
        split_at = len(sequences) - eval_size
        return (
            sequences[:split_at],
            labels[:split_at],
            sequences[split_at:],
            labels[split_at:],
        )

    def _predict_probabilities(self, model, sequences: np.ndarray) -> np.ndarray:
        if len(sequences) == 0:
            return np.asarray([], dtype=np.float64)
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(torch.tensor(sequences, dtype=torch.float32, device=device))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        return probabilities

    def _build_train_sequences(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_window: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sequence_count = len(feature_values) - lookback_window + 1
        if sequence_count <= 0:
            return (
                np.empty((0, lookback_window, feature_values.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        sequences = np.stack(
            [feature_values[offset : offset + lookback_window] for offset in range(sequence_count)],
            axis=0,
        ).astype(np.float32)
        labels = target_values[lookback_window - 1 :].astype(np.float32)
        return sequences, labels

    def _build_segment_sequences(
        self,
        *,
        context_frame: pd.DataFrame,
        segment_frame: pd.DataFrame,
        lookback_window: int,
    ) -> tuple[np.ndarray, pd.Index]:
        combined = pd.concat([context_frame.tail(max(lookback_window - 1, 0)), segment_frame], axis=0)
        if len(combined) < lookback_window or segment_frame.empty:
            return np.empty((0, lookback_window, segment_frame.shape[1]), dtype=np.float32), pd.Index([])

        combined_values = combined.to_numpy(dtype=np.float32)
        sequences = []
        scored_indices = []
        context_rows = len(combined) - len(segment_frame)
        for combined_end in range(lookback_window - 1, len(combined)):
            if combined_end < context_rows:
                continue
            sequences.append(combined_values[combined_end - lookback_window + 1 : combined_end + 1])
            scored_indices.append(combined.index[combined_end])
        if not sequences:
            return np.empty((0, lookback_window, segment_frame.shape[1]), dtype=np.float32), pd.Index([])
        return np.stack(sequences, axis=0).astype(np.float32), pd.Index(scored_indices)

    def _resolved_config(self) -> Dict[str, Any]:
        raw_params = dict(self.model_params or {})
        configured_lookback = (
            self.config.get("ai_model", {}).get("lookback_periods")
            if isinstance(self.config, dict)
            else None
        )
        activation = str(raw_params.get("activation") or "gelu").strip().lower()
        if activation not in {"gelu", "relu"}:
            raise ValueError(f"Unsupported LSTM v2 activation: {activation}")
        feature_mode = str(raw_params.get("feature_mode") or "engineered").strip().lower()
        if feature_mode not in {"engineered", "raw_market", "combined"}:
            raise ValueError(f"Unsupported LSTM v2 feature_mode: {feature_mode}")
        device_info = self.resolve_training_device(str(raw_params.get("device") or "auto"))
        return {
            "architecture_name": "lstm_v2_dense_head",
            "feature_mode": feature_mode,
            "lookback_window": max(int(raw_params.get("lookback_window") or configured_lookback or 20), 1),
            "hidden_size": max(int(raw_params.get("hidden_size") or 32), 4),
            "num_layers": max(int(raw_params.get("num_layers") or 1), 1),
            "dropout": float(raw_params.get("dropout") or 0.0),
            "dense_hidden_size": max(int(raw_params.get("dense_hidden_size") or raw_params.get("hidden_size") or 32), 4),
            "dense_dropout": float(raw_params.get("dense_dropout") or 0.0),
            "activation": activation,
            "bidirectional": bool(raw_params.get("bidirectional", False)),
            "weight_decay": float(raw_params.get("weight_decay") or 0.0),
            "learning_rate": float(raw_params.get("learning_rate") or 1e-3),
            "batch_size": max(int(raw_params.get("batch_size") or 32), 1),
            "epochs": max(int(raw_params.get("epochs") or 25), 1),
            "class_weighting": bool(raw_params.get("class_weighting", True)),
            "early_stopping_patience": max(int(raw_params.get("early_stopping_patience") or 5), 1),
            "decision_threshold": float(raw_params.get("decision_threshold") or 0.5),
            "seed": max(int(raw_params.get("seed") or 42), 0),
            "device": device_info["requested_device"],
            "device_info": device_info,
        }

    def _build_sequence_frame(
        self,
        feature_frame: pd.DataFrame,
        *,
        feature_mode: str,
        engineered_feature_columns: List[str],
        raw_column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        return _build_lstm_sequence_feature_frame(
            feature_frame,
            feature_mode=feature_mode,
            engineered_feature_columns=engineered_feature_columns,
            raw_column_mapping=raw_column_mapping,
        )

    def _resolve_engineered_feature_columns(self, feature_frame: pd.DataFrame, feature_mode: str) -> List[str]:
        if str(feature_mode) == "raw_market":
            return []
        if str(feature_mode) == "engineered":
            return [str(column) for column in feature_frame.columns]
        raw_source_columns = set(self._resolve_raw_column_mapping(feature_frame, "combined").values())
        return [str(column) for column in feature_frame.columns if str(column) not in raw_source_columns]

    def _resolve_raw_column_mapping(self, feature_frame: pd.DataFrame, feature_mode: str) -> Dict[str, str]:
        if str(feature_mode) == "engineered":
            return {}
        normalized = {str(column).strip().lower(): str(column) for column in feature_frame.columns}
        aliases = {
            "open": ["open"],
            "high": ["high"],
            "low": ["low"],
            "close": ["close", "adj close"],
            "volume": ["volume", "tick_volume", "real_volume"],
        }
        mapping: Dict[str, str] = {}
        missing: List[str] = []
        for canonical_name, candidates in aliases.items():
            resolved = next((normalized[candidate] for candidate in candidates if candidate in normalized), "")
            if resolved:
                mapping[canonical_name] = resolved
            else:
                missing.append(canonical_name)
        if missing:
            raise ValueError(f"LSTM v2 {feature_mode} mode requires raw market columns: {', '.join(missing)}")
        return mapping

    @staticmethod
    def _dense_head_summary(config: Dict[str, Any]) -> str:
        direction = "bidirectional" if bool(config["bidirectional"]) else "unidirectional"
        return (
            f"{direction} dense head: {int(config['dense_hidden_size'])} "
            f"{config['activation']} dropout {float(config['dense_dropout']):.2f}"
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(converted):
            return None
        return converted
