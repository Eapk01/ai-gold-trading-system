"""Fold-safe feature-selection helpers for Stage 3 research runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import pandas as pd


@dataclass(frozen=True)
class FeatureSelectorResult:
    """Research-facing output for one fold-local feature-selection pass."""

    selector_name: str
    selected_columns: List[str] = field(default_factory=list)
    ranking_rows: List[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FullSetFeatureSelector:
    """Keep the full candidate feature set for the current fold."""

    def select(self, feature_frame: pd.DataFrame, target: pd.Series) -> FeatureSelectorResult:
        columns = [str(column) for column in feature_frame.columns]
        ranking_rows = [
            {
                "column": column,
                "rank": index + 1,
                "score": None,
            }
            for index, column in enumerate(columns)
        ]
        return FeatureSelectorResult(
            selector_name="full_set",
            selected_columns=columns,
            ranking_rows=ranking_rows,
        )


@dataclass(frozen=True)
class CorrelationFeatureSelector:
    """Select the strongest numeric features by absolute correlation to the target."""

    max_features: int = 30

    def select(self, feature_frame: pd.DataFrame, target: pd.Series) -> FeatureSelectorResult:
        numeric_features = feature_frame.apply(pd.to_numeric, errors="coerce")
        numeric_target = pd.to_numeric(target, errors="coerce")
        correlations = numeric_features.corrwith(numeric_target).abs().dropna().sort_values(ascending=False)
        ranked = correlations.head(int(self.max_features))
        return FeatureSelectorResult(
            selector_name="correlation",
            selected_columns=ranked.index.tolist(),
            ranking_rows=[
                {
                    "column": str(column),
                    "rank": index + 1,
                    "score": float(score),
                }
                for index, (column, score) in enumerate(ranked.items())
            ],
        )


@dataclass(frozen=True)
class VarianceFeatureSelector:
    """Select the highest-variance numeric features on the train fold."""

    max_features: int = 30

    def select(self, feature_frame: pd.DataFrame, target: pd.Series) -> FeatureSelectorResult:
        del target
        numeric_features = feature_frame.apply(pd.to_numeric, errors="coerce")
        variances = numeric_features.var(numeric_only=True).dropna().sort_values(ascending=False)
        ranked = variances.head(int(self.max_features))
        return FeatureSelectorResult(
            selector_name="variance",
            selected_columns=ranked.index.tolist(),
            ranking_rows=[
                {
                    "column": str(column),
                    "rank": index + 1,
                    "score": float(score),
                }
                for index, (column, score) in enumerate(ranked.items())
            ],
        )


def build_feature_selector(selector_name: str, *, max_features: int = 30):
    """Build one supported fold-local feature selector by name."""
    normalized = str(selector_name or "correlation").strip().lower()
    if normalized in {"correlation", "corr"}:
        return CorrelationFeatureSelector(max_features=max_features)
    if normalized in {"variance", "var"}:
        return VarianceFeatureSelector(max_features=max_features)
    if normalized in {"full", "full_set", "none"}:
        return FullSetFeatureSelector()
    raise ValueError(f"Unsupported Stage 3 feature selector: {selector_name}")
