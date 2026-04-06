"""Base trainer contract for research experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.research.schemas import CandidateArtifact, TrainerOutput


class ResearchTrainer(ABC):
    """Abstract trainer interface used by the research pipeline."""

    @abstractmethod
    def fit_predict(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
    ) -> TrainerOutput:
        """Train on the provided fold and return standardized predictions."""
        raise NotImplementedError

    def fit_predict_segments(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        segments: dict[str, pd.DataFrame],
    ) -> dict[str, TrainerOutput]:
        """Train once and predict multiple named segments."""
        return {
            name: self.fit_predict(train_features, train_target, segment_features)
            for name, segment_features in segments.items()
        }

    @abstractmethod
    def fit_candidate_artifact(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        artifact_path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateArtifact:
        """Train a final candidate artifact on the provided dataset."""
        raise NotImplementedError
