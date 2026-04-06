"""Shared research defaults and config-backed resolution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List


DEFAULT_RUNTIME_TARGET_COLUMN = "Future_Direction_1"
DEFAULT_THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
DEFAULT_BASELINE_NAMES = ["majority_class", "persistence"]


@dataclass(frozen=True)
class CommonResearchDefaults:
    """Common request defaults shared across research workflows."""

    baseline_names: List[str] = field(default_factory=list)
    threshold_list: List[float] = field(default_factory=list)
    selector_name: str = "correlation"
    selector_max_features: int = 20
    train_fraction: float = 0.50
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    expanding_window: bool = True


@dataclass(frozen=True)
class Stage12Defaults:
    """Stage 1/2 research defaults."""

    fixed_feature_set_name: str = "baseline_core"


@dataclass(frozen=True)
class Stage3Defaults:
    """Stage 3 feature-study defaults."""

    target_ids: List[str] = field(default_factory=list)
    feature_set_names: List[str] = field(default_factory=list)
    compare_legacy_target: bool = True


@dataclass(frozen=True)
class Stage4Defaults:
    """Stage 4 training defaults."""

    working_target_id: str = "return_threshold_h3_0_05pct"
    feature_set_name: str = "volatility"
    comparison_feature_set_name: str = "baseline_core"


@dataclass(frozen=True)
class Stage5Defaults:
    """Stage 5 automated-search defaults."""

    target_ids: List[str] = field(default_factory=list)
    feature_set_names: List[str] = field(default_factory=list)
    preset_names: List[str] = field(default_factory=list)
    max_worker_cap: int = 4
    min_auto_workers: int = 2


@dataclass(frozen=True)
class TruthGateDefaults:
    """Truth-gate rule defaults."""

    minimum_test_coverage: float = 0.20
    max_validation_test_f1_drift: float = 0.10


@dataclass(frozen=True)
class ResearchDefaults:
    """Resolved research defaults snapshot."""

    runtime_target_column: str
    common: CommonResearchDefaults
    stage12: Stage12Defaults
    stage3: Stage3Defaults
    stage4: Stage4Defaults
    stage5: Stage5Defaults
    truth_gate: TruthGateDefaults

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def default_baseline_names() -> List[str]:
    """Return the canonical default baseline list."""
    return list(DEFAULT_BASELINE_NAMES)


def default_threshold_list() -> List[float]:
    """Return the canonical default threshold grid."""
    return list(DEFAULT_THRESHOLD_GRID)


def get_builtin_research_defaults(
    runtime_target_column: str = DEFAULT_RUNTIME_TARGET_COLUMN,
) -> ResearchDefaults:
    """Return the built-in defaults without applying user config overrides."""
    runtime_target = str(runtime_target_column or DEFAULT_RUNTIME_TARGET_COLUMN).strip() or DEFAULT_RUNTIME_TARGET_COLUMN
    legacy_target_id = _legacy_runtime_target_id(runtime_target)
    return ResearchDefaults(
        runtime_target_column=runtime_target,
        common=CommonResearchDefaults(
            baseline_names=default_baseline_names(),
            threshold_list=default_threshold_list(),
            selector_name="correlation",
            selector_max_features=20,
            train_fraction=0.50,
            validation_fraction=0.10,
            test_fraction=0.10,
            expanding_window=True,
        ),
        stage12=Stage12Defaults(fixed_feature_set_name="baseline_core"),
        stage3=Stage3Defaults(
            target_ids=[
                "return_threshold_h3_0_05pct",
                legacy_target_id,
            ],
            feature_set_names=[
                "baseline_core",
                "momentum",
                "volatility",
                "context",
                "lag_statistical",
                "all_eligible",
            ],
            compare_legacy_target=True,
        ),
        stage4=Stage4Defaults(
            working_target_id="return_threshold_h3_0_05pct",
            feature_set_name="volatility",
            comparison_feature_set_name="baseline_core",
        ),
        stage5=Stage5Defaults(
            target_ids=[
                "return_threshold_h3_0_05pct",
                legacy_target_id,
                "vol_adjusted_h3_x1_0",
            ],
            feature_set_names=["baseline_core", "all_eligible"],
            preset_names=["conservative", "balanced"],
            max_worker_cap=4,
            min_auto_workers=2,
        ),
        truth_gate=TruthGateDefaults(
            minimum_test_coverage=0.20,
            max_validation_test_f1_drift=0.10,
        ),
    )


def resolve_research_defaults(config: Dict[str, Any] | None) -> ResearchDefaults:
    """Resolve research defaults from config with validation and compatibility handling."""
    config_payload = dict(config or {})
    runtime_target_column = str(
        ((config_payload.get("ai_model", {}) or {}).get("target_column") or DEFAULT_RUNTIME_TARGET_COLUMN)
    ).strip() or DEFAULT_RUNTIME_TARGET_COLUMN
    defaults = get_builtin_research_defaults(runtime_target_column)

    research_config = dict((config_payload.get("research", {}) or {}))
    defaults_override = dict((research_config.get("defaults", {}) or {}))
    legacy_stage5_override = dict((research_config.get("stage5_defaults", {}) or {}))
    if legacy_stage5_override:
        stage5_override = dict(defaults_override.get("stage5") or {})
        alias_map = {
            "target_ids": "target_ids",
            "feature_sets": "feature_set_names",
            "presets": "preset_names",
        }
        for legacy_key, stage5_key in alias_map.items():
            if legacy_stage5_override.get(legacy_key) is None:
                continue
            if (
                stage5_override.get(stage5_key) is None
                or stage5_override.get(stage5_key) == getattr(defaults.stage5, stage5_key)
            ):
                stage5_override[stage5_key] = legacy_stage5_override.get(legacy_key)
        defaults_override["stage5"] = stage5_override

    common_override = dict(defaults_override.get("common") or {})
    stage12_override = dict(defaults_override.get("stage12") or {})
    stage3_override = dict(defaults_override.get("stage3") or {})
    stage4_override = dict(defaults_override.get("stage4") or {})
    stage5_override = dict(defaults_override.get("stage5") or {})
    truth_gate_override = dict(defaults_override.get("truth_gate") or {})

    resolved = ResearchDefaults(
        runtime_target_column=runtime_target_column,
        common=CommonResearchDefaults(
            baseline_names=_resolve_string_list(
                common_override.get("baseline_names"),
                defaults.common.baseline_names,
                allow_empty=False,
                dotted_key="research.defaults.common.baseline_names",
            ),
            threshold_list=_resolve_threshold_list(
                common_override.get("threshold_list"),
                defaults.common.threshold_list,
                dotted_key="research.defaults.common.threshold_list",
            ),
            selector_name=_resolve_string(
                common_override.get("selector_name"),
                defaults.common.selector_name,
                dotted_key="research.defaults.common.selector_name",
            ),
            selector_max_features=_resolve_positive_int(
                common_override.get("selector_max_features"),
                defaults.common.selector_max_features,
                dotted_key="research.defaults.common.selector_max_features",
            ),
            train_fraction=_resolve_fraction(
                common_override.get("train_fraction"),
                defaults.common.train_fraction,
                dotted_key="research.defaults.common.train_fraction",
            ),
            validation_fraction=_resolve_fraction(
                common_override.get("validation_fraction"),
                defaults.common.validation_fraction,
                dotted_key="research.defaults.common.validation_fraction",
            ),
            test_fraction=_resolve_fraction(
                common_override.get("test_fraction"),
                defaults.common.test_fraction,
                dotted_key="research.defaults.common.test_fraction",
            ),
            expanding_window=_resolve_bool(
                common_override.get("expanding_window"),
                defaults.common.expanding_window,
                dotted_key="research.defaults.common.expanding_window",
            ),
        ),
        stage12=Stage12Defaults(
            fixed_feature_set_name=_resolve_string(
                stage12_override.get("fixed_feature_set_name"),
                defaults.stage12.fixed_feature_set_name,
                dotted_key="research.defaults.stage12.fixed_feature_set_name",
            ),
        ),
        stage3=Stage3Defaults(
            target_ids=_resolve_string_list(
                stage3_override.get("target_ids"),
                defaults.stage3.target_ids,
                allow_empty=False,
                dotted_key="research.defaults.stage3.target_ids",
            ),
            feature_set_names=_resolve_string_list(
                stage3_override.get("feature_set_names"),
                defaults.stage3.feature_set_names,
                allow_empty=False,
                dotted_key="research.defaults.stage3.feature_set_names",
            ),
            compare_legacy_target=_resolve_bool(
                stage3_override.get("compare_legacy_target"),
                defaults.stage3.compare_legacy_target,
                dotted_key="research.defaults.stage3.compare_legacy_target",
            ),
        ),
        stage4=Stage4Defaults(
            working_target_id=_resolve_string(
                stage4_override.get("working_target_id"),
                defaults.stage4.working_target_id,
                dotted_key="research.defaults.stage4.working_target_id",
            ),
            feature_set_name=_resolve_string(
                stage4_override.get("feature_set_name"),
                defaults.stage4.feature_set_name,
                dotted_key="research.defaults.stage4.feature_set_name",
            ),
            comparison_feature_set_name=_resolve_string(
                stage4_override.get("comparison_feature_set_name"),
                defaults.stage4.comparison_feature_set_name,
                dotted_key="research.defaults.stage4.comparison_feature_set_name",
            ),
        ),
        stage5=Stage5Defaults(
            target_ids=_resolve_string_list(
                stage5_override.get("target_ids"),
                defaults.stage5.target_ids,
                allow_empty=False,
                dotted_key="research.defaults.stage5.target_ids",
            ),
            feature_set_names=_resolve_string_list(
                stage5_override.get("feature_set_names"),
                defaults.stage5.feature_set_names,
                allow_empty=False,
                dotted_key="research.defaults.stage5.feature_set_names",
            ),
            preset_names=_resolve_string_list(
                stage5_override.get("preset_names"),
                defaults.stage5.preset_names,
                allow_empty=False,
                dotted_key="research.defaults.stage5.preset_names",
            ),
            max_worker_cap=_resolve_positive_int(
                stage5_override.get("max_worker_cap"),
                defaults.stage5.max_worker_cap,
                dotted_key="research.defaults.stage5.max_worker_cap",
            ),
            min_auto_workers=_resolve_positive_int(
                stage5_override.get("min_auto_workers"),
                defaults.stage5.min_auto_workers,
                dotted_key="research.defaults.stage5.min_auto_workers",
            ),
        ),
        truth_gate=TruthGateDefaults(
            minimum_test_coverage=_resolve_fraction(
                truth_gate_override.get("minimum_test_coverage"),
                defaults.truth_gate.minimum_test_coverage,
                dotted_key="research.defaults.truth_gate.minimum_test_coverage",
                exclusive_min=False,
            ),
            max_validation_test_f1_drift=_resolve_fraction(
                truth_gate_override.get("max_validation_test_f1_drift"),
                defaults.truth_gate.max_validation_test_f1_drift,
                dotted_key="research.defaults.truth_gate.max_validation_test_f1_drift",
                exclusive_min=False,
            ),
        ),
    )
    _validate_split_fractions(resolved.common)
    _validate_worker_range(resolved.stage5)
    return resolved


def _legacy_runtime_target_id(runtime_target_column: str) -> str:
    return f"legacy_{str(runtime_target_column or DEFAULT_RUNTIME_TARGET_COLUMN).lower()}"


def _resolve_string(value: Any, default: str, *, dotted_key: str) -> str:
    if value is None:
        return str(default)
    candidate = str(value).strip()
    if not candidate:
        raise ValueError(f"{dotted_key} must be a non-empty string")
    return candidate


def _resolve_string_list(
    value: Any,
    default: Iterable[str],
    *,
    allow_empty: bool,
    dotted_key: str,
) -> List[str]:
    if value is None:
        return [str(item) for item in default]
    if not isinstance(value, list):
        raise ValueError(f"{dotted_key} must be a list")
    resolved = [str(item).strip() for item in value]
    if any(not item for item in resolved):
        raise ValueError(f"{dotted_key} must contain only non-empty strings")
    if not allow_empty and not resolved:
        raise ValueError(f"{dotted_key} must be a non-empty list")
    return resolved


def _resolve_threshold_list(value: Any, default: Iterable[float], *, dotted_key: str) -> List[float]:
    if value is None:
        return [float(item) for item in default]
    if not isinstance(value, list):
        raise ValueError(f"{dotted_key} must be a list")
    if not value:
        raise ValueError(f"{dotted_key} must be a non-empty list")
    resolved: List[float] = []
    for item in value:
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{dotted_key} must contain only numeric threshold values") from exc
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{dotted_key} values must be between 0 and 1")
        resolved.append(float(numeric))
    return resolved


def _resolve_positive_int(value: Any, default: int, *, dotted_key: str) -> int:
    if value is None:
        return int(default)
    if not isinstance(value, int):
        raise ValueError(f"{dotted_key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{dotted_key} must be a positive integer")
    return int(value)


def _resolve_fraction(
    value: Any,
    default: float,
    *,
    dotted_key: str,
    exclusive_min: bool = True,
) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_key} must be numeric") from exc
    minimum = 0.0 if not exclusive_min else 0.0
    if exclusive_min and numeric <= minimum:
        raise ValueError(f"{dotted_key} must be > 0")
    if not exclusive_min and numeric < minimum:
        raise ValueError(f"{dotted_key} must be >= 0")
    if numeric > 1.0:
        raise ValueError(f"{dotted_key} must be <= 1")
    return float(numeric)


def _resolve_bool(value: Any, default: bool, *, dotted_key: str) -> bool:
    if value is None:
        return bool(default)
    if not isinstance(value, bool):
        raise ValueError(f"{dotted_key} must be a boolean")
    return bool(value)


def _validate_split_fractions(common: CommonResearchDefaults) -> None:
    total_fraction = float(common.train_fraction + common.validation_fraction + common.test_fraction)
    if total_fraction > 1.0:
        raise ValueError("research.defaults.common split fractions must sum to <= 1.0")


def _validate_worker_range(stage5: Stage5Defaults) -> None:
    if stage5.min_auto_workers > stage5.max_worker_cap:
        raise ValueError(
            "research.defaults.stage5.min_auto_workers must be <= research.defaults.stage5.max_worker_cap"
        )
