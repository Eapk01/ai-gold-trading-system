"""Feature-study orchestration helpers for Stage 3 research runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd

from .feature_sets import (
    build_feature_inventory,
    enrich_inventory_with_named_sets,
    get_group_display_name,
    resolve_feature_sets,
)
from .labels import BinaryTargetSpec, spec_display_name
from .schemas import (
    FeatureStudyFoldSelection,
    FeatureStudyRequest,
    FeatureStudyResult,
    FeatureStudySetResult,
)
from .target_study import TargetStudyRunner, build_target_result_stub


ExperimentExecutor = Callable[..., Dict[str, Any]]


@dataclass
class FeatureStudyRunner:
    """Run Stage 3 feature studies by wrapping the Stage 1 experiment flow."""

    def run(
        self,
        *,
        request: FeatureStudyRequest,
        feature_frame: pd.DataFrame,
        target_specs: Sequence[BinaryTargetSpec],
        experiment_executor: ExperimentExecutor,
        artifact_timestamp: str,
    ) -> FeatureStudyResult:
        feature_sets = resolve_feature_sets(feature_frame.columns, request.feature_set_names)
        inventory_rows = enrich_inventory_with_named_sets(
            build_feature_inventory(feature_frame.columns),
            feature_sets,
        )

        target_runner = TargetStudyRunner()
        materialized_targets = target_runner.materialize_targets(feature_frame, list(target_specs))
        working_target_id = request.target_specs[0].get("spec_id", "") if request.target_specs else ""
        set_results: List[FeatureStudySetResult] = []

        for materialized in materialized_targets:
            target_stub = build_target_result_stub(materialized)
            for feature_set in feature_sets:
                set_result = FeatureStudySetResult(
                    target_id=target_stub.target_id,
                    target_display_name=target_stub.display_name,
                    feature_set_name=feature_set.name,
                    feature_set_display_name=feature_set.display_name,
                    candidate_feature_count=len(feature_set.columns),
                    candidate_columns=list(feature_set.columns),
                )
                if not feature_set.columns:
                    set_result.error = "No eligible columns resolved for this feature set"
                    set_results.append(set_result)
                    continue

                try:
                    execution = experiment_executor(
                        target_id=target_stub.target_id,
                        target_series=materialized.target,
                        target_spec=materialized.spec,
                        feature_set=feature_set,
                        artifact_prefix=f"research_experiment_{artifact_timestamp}_{target_stub.target_id}_{feature_set.name}",
                        selector_name=request.selector_name,
                        selector_max_features=request.selector_max_features,
                        request=request,
                    )
                    experiment_result = execution["result"]
                    set_result.experiment_report_file = execution["artifacts"]["report_file"]
                    set_result.experiment_summary = execution["summary"]
                    set_result.aggregate_metrics = experiment_result.aggregate_metrics
                    set_result.baseline_comparison = experiment_result.baseline_comparison
                    set_result.fold_selections = self._build_fold_selections(
                        target_id=target_stub.target_id,
                        target_display_name=target_stub.display_name,
                        feature_set_name=feature_set.name,
                        raw_selection_rows=experiment_result.metadata.get("fold_feature_selections", []),
                    )
                    set_result.stability_rows = self._build_feature_stability_rows(
                        feature_set_name=feature_set.name,
                        fold_selections=set_result.fold_selections,
                        inventory_rows=inventory_rows,
                    )
                    set_result.group_stability_rows = self._build_group_stability_rows(
                        feature_set_name=feature_set.name,
                        fold_selections=set_result.fold_selections,
                        inventory_rows=inventory_rows,
                        folds=experiment_result.folds,
                    )
                except Exception as exc:
                    set_result.error = str(exc)
                set_results.append(set_result)

        comparison_rows = self.build_comparison_rows(set_results)
        return FeatureStudyResult(
            study_name=request.study_name,
            working_target_id=working_target_id,
            request={
                "target_specs": list(request.target_specs),
                "feature_set_names": list(request.feature_set_names),
                "selector_name": request.selector_name,
                "selector_max_features": request.selector_max_features,
                "trainer_name": request.trainer_name,
                "baseline_names": list(request.baseline_names),
                "train_size": request.train_size,
                "validation_size": request.validation_size,
                "test_size": request.test_size,
                "step_size": request.step_size,
                "threshold_list": list(request.threshold_list),
                "expanding_window": request.expanding_window,
                "compare_legacy_target": request.compare_legacy_target,
            },
            inventory_rows=inventory_rows,
            set_results=set_results,
            comparison_rows=comparison_rows,
            metadata={
                "working_target_display_name": spec_display_name(target_specs[0]) if target_specs else "",
                "runtime_feature_pipeline_unchanged": True,
                "runtime_target_unchanged": True,
                "deferred_items": ["runtime_feature_generator_refactor"],
            },
        )

    def build_comparison_rows(self, set_results: Sequence[FeatureStudySetResult]) -> List[Dict[str, Any]]:
        """Flatten Stage 3 set results into a comparison table."""
        rows: List[Dict[str, Any]] = []
        for result in set_results:
            baseline_metrics = result.baseline_comparison.get("baselines", {}) if result.baseline_comparison else {}
            best_baseline_name = ""
            best_baseline_accuracy = None
            for name, metrics in baseline_metrics.items():
                accuracy = metrics.get("mean_test_accuracy")
                if best_baseline_accuracy is None or (accuracy is not None and accuracy > best_baseline_accuracy):
                    best_baseline_name = name
                    best_baseline_accuracy = accuracy

            rows.append(
                {
                    "target_id": result.target_id,
                    "target_display_name": result.target_display_name,
                    "feature_set_name": result.feature_set_name,
                    "feature_set_display_name": result.feature_set_display_name,
                    "candidate_feature_count": result.candidate_feature_count,
                    "mean_test_accuracy": result.aggregate_metrics.get("mean_test_accuracy"),
                    "best_baseline_name": best_baseline_name,
                    "best_baseline_accuracy": best_baseline_accuracy,
                    "stable_feature_count": len([row for row in result.stability_rows if row.get("selection_rate", 0.0) >= 0.5]),
                    "error": result.error,
                }
            )
        return rows

    def build_inventory_frame(self, result: FeatureStudyResult) -> pd.DataFrame:
        """Return the feature inventory as a dataframe suitable for CSV export."""
        return pd.DataFrame(
            [
                {
                    "column": row.column,
                    "group": row.group,
                    "group_display_name": get_group_display_name(row.group),
                    "eligible": row.eligible,
                    "included_in_named_sets": ",".join(row.included_in_named_sets),
                    "exclusion_reason": row.exclusion_reason,
                }
                for row in result.inventory_rows
            ]
        )

    def build_fold_selection_frame(self, result: FeatureStudyResult) -> pd.DataFrame:
        """Return one row per selected feature per fold."""
        rows: List[Dict[str, Any]] = []
        for set_result in result.set_results:
            for fold_selection in set_result.fold_selections:
                rank_by_column = {
                    ranking_row.get("column"): ranking_row
                    for ranking_row in fold_selection.ranking_rows
                }
                for column in fold_selection.selected_columns:
                    ranking_row = rank_by_column.get(column, {})
                    rows.append(
                        {
                            "target_id": fold_selection.target_id,
                            "target_display_name": fold_selection.target_display_name,
                            "feature_set_name": fold_selection.feature_set_name,
                            "fold_name": fold_selection.fold_name,
                            "selector_name": fold_selection.selector_name,
                            "candidate_count": fold_selection.candidate_count,
                            "selected_count": fold_selection.selected_count,
                            "column": column,
                            "rank": ranking_row.get("rank"),
                            "score": ranking_row.get("score"),
                        }
                    )
        return pd.DataFrame(rows)

    def build_stability_frame(self, result: FeatureStudyResult) -> pd.DataFrame:
        """Return combined feature-level and group-level stability rows."""
        rows: List[Dict[str, Any]] = []
        for set_result in result.set_results:
            for row in set_result.stability_rows:
                rows.append({"row_type": "feature", **row})
            for row in set_result.group_stability_rows:
                rows.append({"row_type": "group", **row})
        return pd.DataFrame(rows)

    def _build_fold_selections(
        self,
        *,
        target_id: str,
        target_display_name: str,
        feature_set_name: str,
        raw_selection_rows: Sequence[Dict[str, Any]],
    ) -> List[FeatureStudyFoldSelection]:
        return [
            FeatureStudyFoldSelection(
                target_id=target_id,
                target_display_name=target_display_name,
                feature_set_name=feature_set_name,
                fold_name=row.get("fold_name", ""),
                selector_name=row.get("selector_name", ""),
                candidate_count=int(row.get("candidate_count", 0)),
                selected_count=int(row.get("selected_count", 0)),
                selected_columns=list(row.get("selected_columns", [])),
                ranking_rows=list(row.get("ranking_rows", [])),
            )
            for row in raw_selection_rows
        ]

    def _build_feature_stability_rows(
        self,
        *,
        feature_set_name: str,
        fold_selections: Sequence[FeatureStudyFoldSelection],
        inventory_rows: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        if not fold_selections:
            return []

        group_by_column = {row.column: row.group for row in inventory_rows}
        fold_count = len(fold_selections)
        counts: Dict[str, Dict[str, Any]] = {}
        for fold_selection in fold_selections:
            rank_by_column = {
                ranking_row.get("column"): ranking_row
                for ranking_row in fold_selection.ranking_rows
            }
            for column in fold_selection.selected_columns:
                bucket = counts.setdefault(
                    column,
                    {
                        "count": 0,
                        "ranks": [],
                        "scores": [],
                    },
                )
                bucket["count"] += 1
                ranking_row = rank_by_column.get(column, {})
                if ranking_row.get("rank") is not None:
                    bucket["ranks"].append(float(ranking_row["rank"]))
                if ranking_row.get("score") is not None:
                    bucket["scores"].append(float(ranking_row["score"]))

        rows = []
        for column, stats in sorted(counts.items()):
            rows.append(
                {
                    "feature_set_name": feature_set_name,
                    "column": column,
                    "group": group_by_column.get(column, "excluded"),
                    "group_display_name": get_group_display_name(group_by_column.get(column, "excluded")),
                    "selected_in_folds": int(stats["count"]),
                    "fold_count": int(fold_count),
                    "selection_rate": float(stats["count"] / fold_count) if fold_count else 0.0,
                    "average_rank": float(sum(stats["ranks"]) / len(stats["ranks"])) if stats["ranks"] else None,
                    "average_score": float(sum(stats["scores"]) / len(stats["scores"])) if stats["scores"] else None,
                }
            )
        return rows

    def _build_group_stability_rows(
        self,
        *,
        feature_set_name: str,
        fold_selections: Sequence[FeatureStudyFoldSelection],
        inventory_rows: Sequence[Any],
        folds: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        if not fold_selections:
            return []

        group_by_column = {row.column: row.group for row in inventory_rows}
        group_counts: Dict[str, Dict[str, Any]] = {}
        for fold_selection in fold_selections:
            seen_groups = set()
            for column in fold_selection.selected_columns:
                group_name = group_by_column.get(column, "excluded")
                bucket = group_counts.setdefault(group_name, {"selected_features": 0, "fold_presence": 0})
                bucket["selected_features"] += 1
                if group_name not in seen_groups:
                    bucket["fold_presence"] += 1
                    seen_groups.add(group_name)

        test_fold_metrics = [fold.metrics for fold in folds if getattr(fold, "model_name", "") == "current_ensemble" and getattr(fold, "split_segment", "") == "test"]
        accuracies = [float(metrics.get("accuracy") or 0.0) for metrics in test_fold_metrics]
        collapse_threshold = (sum(accuracies) / len(accuracies) * 0.9) if accuracies else 0.0
        weak_fold_count = sum(1 for accuracy in accuracies if accuracy < collapse_threshold) if accuracies else 0
        fold_count = len(fold_selections)

        rows = []
        for group_name, stats in sorted(group_counts.items()):
            rows.append(
                {
                    "feature_set_name": feature_set_name,
                    "group": group_name,
                    "group_display_name": get_group_display_name(group_name),
                    "selected_feature_total": int(stats["selected_features"]),
                    "fold_presence_count": int(stats["fold_presence"]),
                    "fold_presence_rate": float(stats["fold_presence"] / fold_count) if fold_count else 0.0,
                    "performance_collapse_flag": bool(weak_fold_count > 0 and stats["fold_presence"] < fold_count),
                    "weak_test_fold_count": int(weak_fold_count),
                }
            )
        return rows
