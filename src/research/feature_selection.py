"""Fold-safe feature-selection helpers for Feature selection research runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import pandas as pd


def _filter_variable_numeric_columns(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that cannot produce stable correlation statistics."""
    numeric_features = feature_frame.apply(pd.to_numeric, errors="coerce")
    variances = numeric_features.var(numeric_only=True)
    variable_columns = variances[variances > 0].index.tolist()
    return numeric_features.loc[:, variable_columns]


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
        numeric_features = _filter_variable_numeric_columns(feature_frame)
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
        numeric_features = _filter_variable_numeric_columns(feature_frame)
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


@dataclass(frozen=True)
class PassthroughFeatureSelector:
    """Apply a base selector while always keeping required passthrough columns."""

    base_selector: Any
    passthrough_columns: List[str] = field(default_factory=list)

    def select(self, feature_frame: pd.DataFrame, target: pd.Series) -> FeatureSelectorResult:
        passthrough = [column for column in self.passthrough_columns if column in feature_frame.columns]
        selector_columns = [column for column in feature_frame.columns if column not in set(passthrough)]
        if selector_columns:
            base_result = self.base_selector.select(feature_frame.loc[:, selector_columns], target)
            selected_columns = list(base_result.selected_columns) or selector_columns
            ranking_rows = list(base_result.ranking_rows)
            selector_name = str(base_result.selector_name)
        else:
            selected_columns = []
            ranking_rows = []
            selector_name = "passthrough_only"
        final_columns = _dedupe(selected_columns + passthrough)
        return FeatureSelectorResult(
            selector_name=f"{selector_name}_with_passthrough",
            selected_columns=final_columns,
            ranking_rows=ranking_rows + [
                {
                    "column": column,
                    "rank": len(ranking_rows) + index + 1,
                    "score": None,
                    "passthrough": True,
                }
                for index, column in enumerate(passthrough)
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
    raise ValueError(f"Unsupported Feature selection feature selector: {selector_name}")


def _dedupe(columns: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered
