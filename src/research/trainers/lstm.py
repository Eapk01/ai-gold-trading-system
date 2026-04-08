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
from src.runtime_predictor import LSTMSequenceClassifier, build_lstm_artifact_payload, torch
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
    """Train a binary LSTM classifier on fold-local rolling windows."""

    config: Dict
    target_column: str | None = None
    model_params: Dict[str, Any] | None = None

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
            numeric_segment = raw_segment.apply(pd.to_numeric, errors="coerce")
            prepared_segment = prepared["preprocessor"].transform(numeric_segment)
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
        artifact_payload = build_lstm_artifact_payload(
            model=model,
            feature_columns=train_features.columns,
            lookback_window=prepared["lookback_window"],
            fill_values=prepared["preprocessor"].fill_values,
            scaler_mean=np.asarray(prepared["scaler"].mean_, dtype=np.float64),
            scaler_scale=np.asarray(prepared["scaler"].scale_, dtype=np.float64),
            decision_threshold=prepared["decision_threshold"],
            target_column=str(self.target_column or ""),
            model_settings=dict(prepared["model_settings"]),
            metadata={
                "trainer_name": "lstm",
                "selected_features": list(train_features.columns),
                "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
                **dict(metadata or {}),
            },
        )
        joblib.dump(artifact_payload, artifact_path)
        return CandidateArtifact(
            artifact_path=str(artifact_path),
            selected_features=list(train_features.columns),
            trainer_name="lstm",
            metadata={
                "artifact_type": "runtime_predictor",
                "runtime_loader": "lstm",
                "lookback_window": int(prepared["lookback_window"]),
                "preprocessing": "median_imputation_then_standard_scaling_then_lstm_sequence_windowing",
                "model_params": dict(prepared["model_settings"]),
                "target_column": str(self.target_column or ""),
                **dict(metadata or {}),
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
        train_frame = train_features.apply(pd.to_numeric, errors="coerce")
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
            "model_settings": {
                "input_size": int(train_sequences.shape[2]),
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                "dropout": config["dropout"],
            },
        }

    def _train_network(self, sequences: np.ndarray, labels: np.ndarray, config: Dict[str, Any]):
        torch.manual_seed(int(config["seed"]))
        model = LSTMSequenceClassifier(
            input_size=int(sequences.shape[2]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        criterion = self._build_loss(labels, config)

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
            torch.tensor(eval_sequences, dtype=torch.float32),
            torch.tensor(eval_labels, dtype=torch.float32),
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

    def _build_loss(self, labels: np.ndarray, config: Dict[str, Any]):
        if not bool(config["class_weighting"]):
            return nn.BCEWithLogitsLoss()

        positive_count = float(np.sum(labels == 1.0))
        negative_count = float(np.sum(labels == 0.0))
        if positive_count <= 0.0 or negative_count <= 0.0:
            return nn.BCEWithLogitsLoss()

        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
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
        with torch.no_grad():
            logits = model(torch.tensor(sequences, dtype=torch.float32))
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
        return {
            "lookback_window": max(int(raw_params.get("lookback_window") or configured_lookback or 20), 1),
            "hidden_size": max(int(raw_params.get("hidden_size") or 32), 4),
            "num_layers": max(int(raw_params.get("num_layers") or 1), 1),
            "dropout": float(raw_params.get("dropout") or 0.0),
            "learning_rate": float(raw_params.get("learning_rate") or 1e-3),
            "batch_size": max(int(raw_params.get("batch_size") or 32), 1),
            "epochs": max(int(raw_params.get("epochs") or 25), 1),
            "class_weighting": bool(raw_params.get("class_weighting", True)),
            "early_stopping_patience": max(int(raw_params.get("early_stopping_patience") or 5), 1),
            "decision_threshold": float(raw_params.get("decision_threshold") or 0.5),
            "seed": max(int(raw_params.get("seed") or 42), 0),
        }
