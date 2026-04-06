"""Target-study helpers built on top of the Stage 1 experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .labels import (
    BinaryTargetSpec,
    build_binary_target_labels,
    build_observed_persistence_labels,
    serialize_target_spec,
    spec_display_name,
    spec_target_type,
)
from .schemas import TargetStudyTargetResult, TargetSummary


@dataclass
class MaterializedTarget:
    """A concrete target series plus summary metadata."""

    spec: BinaryTargetSpec
    target: pd.Series
    summary: TargetSummary


@dataclass
class TargetStudyRunner:
    """Build, summarize, and compare target definitions on one dataset."""

    def materialize_targets(
        self,
        feature_frame: pd.DataFrame,
        target_specs: Sequence[BinaryTargetSpec],
    ) -> list[MaterializedTarget]:
        """Materialize and summarize target specs deterministically."""
        return [
            MaterializedTarget(
                spec=spec,
                target=target,
                summary=self._summarize_target(spec, target, feature_frame),
            )
            for spec in target_specs
            for target in [build_binary_target_labels(feature_frame, spec)]
        ]

    def build_comparison_rows(self, target_results: Sequence[TargetStudyTargetResult]) -> list[dict]:
        """Flatten target-study results into a table-friendly comparison view."""
        rows: list[dict] = []
        for result in target_results:
            summary = result.summary
            baseline_map = result.baseline_comparison.get("baselines") or {}
            row = {
                "target_id": result.target_id,
                "display_name": result.display_name,
                "target_type": result.target_type,
                "positive_rate": summary.positive_rate,
                "negative_rate": summary.negative_rate,
                "missing_rate": summary.missing_rate,
                "majority_rate": summary.majority_rate,
                "persistence_rate": summary.persistence_rate,
                "model_mean_test_accuracy": result.aggregate_metrics.get("mean_test_accuracy"),
                "majority_class_mean_test_accuracy": (baseline_map.get("majority_class") or {}).get("mean_test_accuracy"),
                "persistence_mean_test_accuracy": (baseline_map.get("persistence") or {}).get("mean_test_accuracy"),
                "experiment_report_file": result.experiment_report_file,
                "error": result.error,
            }
            rows.append(row)
        return rows

    def _summarize_target(self, spec: BinaryTargetSpec, target: pd.Series, feature_frame: pd.DataFrame) -> TargetSummary:
        numeric_target = pd.to_numeric(target, errors="coerce")
        scored_target = numeric_target.dropna()
        total_rows = int(len(target))
        scored_rows = int(len(scored_target))
        positive_count = int((scored_target == 1).sum())
        negative_count = int((scored_target == 0).sum())
        missing_count = total_rows - scored_rows
        majority_class = None
        majority_rate = 0.0
        persistence_rate = None

        if scored_rows:
            majority_class = 1 if positive_count >= negative_count else 0
            majority_rate = max(positive_count, negative_count) / scored_rows
            observed_persistence = pd.to_numeric(build_observed_persistence_labels(feature_frame, spec), errors="coerce")
            valid_persistence = numeric_target.notna() & observed_persistence.notna()
            if bool(valid_persistence.any()):
                persistence_rate = float(
                    (numeric_target.loc[valid_persistence].astype(float) == observed_persistence.loc[valid_persistence].astype(float)).mean()
                )

        return TargetSummary(
            target_id=getattr(spec, "spec_id", spec.__class__.__name__),
            target_type=spec_target_type(spec),
            display_name=spec_display_name(spec),
            total_rows=total_rows,
            scored_rows=scored_rows,
            positive_count=positive_count,
            negative_count=negative_count,
            missing_count=missing_count,
            positive_rate=(positive_count / scored_rows) if scored_rows else 0.0,
            negative_rate=(negative_count / scored_rows) if scored_rows else 0.0,
            missing_rate=(missing_count / total_rows) if total_rows else 0.0,
            majority_class=majority_class,
            majority_rate=float(majority_rate),
            persistence_rate=persistence_rate,
        )


def build_target_result_stub(materialized: MaterializedTarget) -> TargetStudyTargetResult:
    """Create a result stub for one materialized target."""
    return TargetStudyTargetResult(
        target_id=materialized.summary.target_id,
        target_type=materialized.summary.target_type,
        display_name=materialized.summary.display_name,
        spec=serialize_target_spec(materialized.spec),
        summary=materialized.summary,
    )
