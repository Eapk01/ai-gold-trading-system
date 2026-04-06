"""Shared dataclasses for research experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .defaults import default_baseline_names, default_threshold_list


@dataclass(frozen=True)
class ResearchSplit:
    """Index boundaries for one time-ordered train/validation/test slice."""

    name: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> Dict[str, int | str]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class ResearchExperimentRequest:
    """Input contract for one Stage 1 research evaluation run."""

    experiment_name: str
    target_column: str
    feature_columns: List[str]
    trainer_name: str = "current_ensemble"
    baseline_names: List[str] = field(default_factory=default_baseline_names)
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0
    step_size: int = 0
    threshold_list: List[float] = field(default_factory=default_threshold_list)
    expanding_window: bool = True

    def normalized_thresholds(self) -> tuple[float, ...]:
        """Return a stable float threshold tuple."""
        return tuple(float(value) for value in self.threshold_list)


@dataclass
class FoldMetrics:
    """Compact metric payload for one fold."""

    fold_name: str
    model_name: str
    split_segment: str
    train_rows: int
    validation_rows: int
    test_rows: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionArtifacts:
    """Saved prediction-level outputs for one fold or experiment."""

    predictions_file: str = ""
    probabilities_file: str = ""
    calibration_file: str = ""
    threshold_metrics_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerOutput:
    """Standard output contract returned by any trainer implementation."""

    prediction: Any
    confidence: Any = None
    probabilities: Any = None
    selected_features: List[str] = field(default_factory=list)
    model_artifact_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateArtifact:
    """Standard artifact output for a fully trained experiment candidate."""

    artifact_path: str
    selected_features: List[str] = field(default_factory=list)
    trainer_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Persistable result for a full experiment run."""

    experiment_name: str
    target_column: str
    feature_columns: List[str]
    request: Dict[str, Any] = field(default_factory=dict)
    fold_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    folds: List[FoldMetrics] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    calibration_summary: Dict[str, Any] = field(default_factory=dict)
    threshold_summary: Dict[str, Any] = field(default_factory=dict)
    prediction_artifacts: List[PredictionArtifacts] = field(default_factory=list)
    integrity: Dict[str, Any] = field(default_factory=dict)
    integrity_artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class TargetStudyRequest:
    """Input contract for a Stage 2 target study run."""

    study_name: str
    feature_columns: List[str]
    target_specs: List[Dict[str, Any]]
    trainer_name: str = "current_ensemble"
    baseline_names: List[str] = field(default_factory=default_baseline_names)
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0
    step_size: int = 0
    threshold_list: List[float] = field(default_factory=default_threshold_list)
    expanding_window: bool = True

    def normalized_thresholds(self) -> tuple[float, ...]:
        """Return a stable float threshold tuple."""
        return tuple(float(value) for value in self.threshold_list)


@dataclass
class TargetSummary:
    """Class balance and difficulty summary for one target definition."""

    target_id: str
    target_type: str
    display_name: str
    total_rows: int
    scored_rows: int
    positive_count: int
    negative_count: int
    missing_count: int
    positive_rate: float
    negative_rate: float
    missing_rate: float
    majority_class: int | None = None
    majority_rate: float = 0.0
    persistence_rate: float | None = None


@dataclass
class TargetStudyTargetResult:
    """Persistable result for one target inside a Stage 2 study."""

    target_id: str
    target_type: str
    display_name: str
    spec: Dict[str, Any]
    summary: TargetSummary
    experiment_report_file: str = ""
    experiment_summary: Dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    integrity: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class TargetStudyResult:
    """Persistable result for a Stage 2 target comparison study."""

    study_name: str
    feature_columns: List[str]
    request: Dict[str, Any] = field(default_factory=dict)
    target_results: List[TargetStudyTargetResult] = field(default_factory=list)
    comparison_rows: List[Dict[str, Any]] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    integrity: Dict[str, Any] = field(default_factory=dict)
    integrity_artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class FeatureStudyRequest:
    """Input contract for a Stage 3 feature study run."""

    study_name: str
    target_specs: List[Dict[str, Any]]
    feature_set_names: List[str]
    selector_name: str = "correlation"
    selector_max_features: int = 20
    trainer_name: str = "current_ensemble"
    baseline_names: List[str] = field(default_factory=default_baseline_names)
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0
    step_size: int = 0
    threshold_list: List[float] = field(default_factory=default_threshold_list)
    expanding_window: bool = True
    compare_legacy_target: bool = True

    def normalized_thresholds(self) -> tuple[float, ...]:
        """Return a stable float threshold tuple."""
        return tuple(float(value) for value in self.threshold_list)


@dataclass
class FeatureInventoryRow:
    """One feature inventory row used in Stage 3 studies."""

    column: str
    group: str
    eligible: bool
    included_in_named_sets: List[str] = field(default_factory=list)
    exclusion_reason: str = ""
    source: str = "prepared_feature_matrix"


@dataclass
class FeatureStudyFoldSelection:
    """Per-fold selected-feature payload for one feature-set run."""

    target_id: str
    target_display_name: str
    feature_set_name: str
    fold_name: str
    selector_name: str
    candidate_count: int
    selected_count: int
    selected_columns: List[str] = field(default_factory=list)
    ranking_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FeatureStudySetResult:
    """Persistable result for one target/feature-set combination."""

    target_id: str
    target_display_name: str
    feature_set_name: str
    feature_set_display_name: str
    candidate_feature_count: int
    candidate_columns: List[str] = field(default_factory=list)
    experiment_report_file: str = ""
    experiment_summary: Dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    fold_selections: List[FeatureStudyFoldSelection] = field(default_factory=list)
    stability_rows: List[Dict[str, Any]] = field(default_factory=list)
    group_stability_rows: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class FeatureStudyResult:
    """Persistable result for a Stage 3 feature comparison study."""

    study_name: str
    working_target_id: str
    request: Dict[str, Any] = field(default_factory=dict)
    inventory_rows: List[FeatureInventoryRow] = field(default_factory=list)
    set_results: List[FeatureStudySetResult] = field(default_factory=list)
    comparison_rows: List[Dict[str, Any]] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingExperimentRequest:
    """Canonical Stage 4 experiment request."""

    experiment_id: str
    experiment_name: str
    target_spec: Dict[str, Any]
    feature_set_name: str
    comparison_feature_set_name: str = ""
    selector_name: str = "correlation"
    selector_max_features: int = 20
    trainer_name: str = "current_ensemble"
    trainer_params: Dict[str, Any] = field(default_factory=dict)
    baseline_names: List[str] = field(default_factory=default_baseline_names)
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0
    step_size: int = 0
    threshold_list: List[float] = field(default_factory=default_threshold_list)
    expanding_window: bool = True

    def normalized_thresholds(self) -> tuple[float, ...]:
        """Return a stable float threshold tuple."""
        return tuple(float(value) for value in self.threshold_list)


@dataclass
class TrainingExperimentResult:
    """Persistable Stage 4 canonical experiment result."""

    experiment_id: str
    experiment_name: str
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    target_spec: Dict[str, Any] = field(default_factory=dict)
    feature_set_name: str = ""
    comparison_feature_set_name: str = ""
    resolved_feature_columns: List[str] = field(default_factory=list)
    selector_name: str = ""
    selector_settings: Dict[str, Any] = field(default_factory=dict)
    trainer_name: str = ""
    trainer_params: Dict[str, Any] = field(default_factory=dict)
    split_settings: Dict[str, Any] = field(default_factory=dict)
    threshold_list: List[float] = field(default_factory=list)
    selected_threshold: float | None = None
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    folds: List[FoldMetrics] = field(default_factory=list)
    prediction_artifacts: List[PredictionArtifacts] = field(default_factory=list)
    candidate_artifact: CandidateArtifact | None = None
    comparison_runs: List[Dict[str, Any]] = field(default_factory=list)
    promotion_status: str = "not_promoted"
    promotion_manifest_file: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    diagnostics_artifact_paths: Dict[str, str] = field(default_factory=dict)
    integrity: Dict[str, Any] = field(default_factory=dict)
    integrity_artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class SearchRequest:
    """Input contract for a bounded Stage 5 automated search run."""

    search_id: str
    search_name: str
    target_spec: Dict[str, Any]
    feature_set_names: List[str]
    target_specs: List[Dict[str, Any]] = field(default_factory=list)
    trainer_name: str = "current_ensemble"
    preset_names: List[str] = field(default_factory=list)
    baseline_names: List[str] = field(default_factory=default_baseline_names)
    selector_name: str = "correlation"
    selector_max_features: int = 20
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0
    step_size: int = 0
    threshold_list: List[float] = field(default_factory=default_threshold_list)
    expanding_window: bool = True
    max_workers: int | None = None
    execution_mode: str = "parallel_candidate_threads"

    def normalized_thresholds(self) -> tuple[float, ...]:
        """Return a stable float threshold tuple."""
        return tuple(float(value) for value in self.threshold_list)

    def resolved_target_specs(self) -> List[Dict[str, Any]]:
        """Return the explicit target list for the search run."""
        if self.target_specs:
            return [dict(spec) for spec in self.target_specs]
        return [dict(self.target_spec)] if self.target_spec else []


@dataclass
class SearchCandidateSummary:
    """Ranking-ready summary for one Stage 5 candidate."""

    candidate_id: str
    experiment_id: str
    experiment_name: str
    trainer_name: str
    feature_set_name: str
    preset_name: str
    target_spec_id: str = ""
    target_display_name: str = ""
    trainer_params: Dict[str, Any] = field(default_factory=dict)
    selected_threshold: float | None = None
    report_file: str = ""
    candidate_artifact_path: str = ""
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    test_summary: Dict[str, Any] = field(default_factory=dict)
    overall_mean_test_accuracy: float | None = None
    majority_baseline_mean_test_accuracy: float | None = None
    expected_fold_count: int = 0
    runtime_feature_contract_ok: bool = True
    rank_tuple: List[float] = field(default_factory=list)
    is_recommended: bool = False
    passed_test_guardrail: bool = False
    passed_truth_gate: bool = False
    truth_gate_failures: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    recommendation_reason: str = ""
    execution_status: str = "completed"
    error_message: str = ""
    elapsed_seconds: float | None = None


@dataclass
class SearchResult:
    """Persistable result for a bounded Stage 5 automated search run."""

    search_id: str
    search_name: str
    target_spec: Dict[str, Any] = field(default_factory=dict)
    target_specs: List[Dict[str, Any]] = field(default_factory=list)
    trainer_name: str = ""
    feature_set_names: List[str] = field(default_factory=list)
    preset_definitions: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    candidates: List[SearchCandidateSummary] = field(default_factory=list)
    leaderboard_rows: List[Dict[str, Any]] = field(default_factory=list)
    recommended_winner: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    integrity: Dict[str, Any] = field(default_factory=dict)
    integrity_artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "sequential"
    resolved_max_workers: int = 1
    successful_candidate_count: int = 0
    failed_candidate_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
