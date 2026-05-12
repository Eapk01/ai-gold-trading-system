"""Shared defaults for the single research-search workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List


DEFAULT_RUNTIME_TARGET_COLUMN = "Future_Direction_1"
DEFAULT_THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
DEFAULT_BASELINE_NAMES = ["majority_class", "persistence"]


@dataclass(frozen=True)
class CommonResearchDefaults:
    """Shared request defaults used by each candidate run."""

    baseline_names: List[str] = field(default_factory=list)
    threshold_list: List[float] = field(default_factory=list)
    selector_name: str = "correlation"
    selector_max_features: int = 20
    train_fraction: float = 0.50
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    expanding_window: bool = True


@dataclass(frozen=True)
class ResearchSearchDefaults:
    """Search-space defaults for the primary research workflow."""

    target_ids: List[str] = field(default_factory=list)
    feature_set_names: List[str] = field(default_factory=list)
    trainer_name: str = "current_ensemble"
    preset_names: List[str] = field(default_factory=list)
    working_target_id: str = "return_threshold_h3_0_05pct"
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
    search: ResearchSearchDefaults
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
    """Return built-in defaults for the primary research-search workflow."""
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
        search=ResearchSearchDefaults(
            target_ids=[
                "return_threshold_h3_0_05pct",
                legacy_target_id,
                "vol_adjusted_h3_x1_0",
            ],
            feature_set_names=["baseline_core", "all_eligible"],
            trainer_name="current_ensemble",
            preset_names=["conservative", "balanced"],
            working_target_id="return_threshold_h3_0_05pct",
            max_worker_cap=4,
            min_auto_workers=2,
        ),
        truth_gate=TruthGateDefaults(
            minimum_test_coverage=0.20,
            max_validation_test_f1_drift=0.10,
        ),
    )


def resolve_research_defaults(config: Dict[str, Any] | None) -> ResearchDefaults:
    """Resolve research-search defaults from config."""
    config_payload = dict(config or {})
    runtime_target_column = str(
        ((config_payload.get("ai_model", {}) or {}).get("target_column") or DEFAULT_RUNTIME_TARGET_COLUMN)
    ).strip() or DEFAULT_RUNTIME_TARGET_COLUMN
    defaults = get_builtin_research_defaults(runtime_target_column)

    research_config = dict((config_payload.get("research", {}) or {}))
    defaults_override = dict((research_config.get("defaults", {}) or {}))
    common_override = dict(defaults_override.get("common") or {})
    search_override = dict(defaults_override.get("search") or {})
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
        search=ResearchSearchDefaults(
            target_ids=_resolve_string_list(
                search_override.get("target_ids"),
                defaults.search.target_ids,
                allow_empty=False,
                dotted_key="research.defaults.search.target_ids",
            ),
            feature_set_names=_resolve_string_list(
                search_override.get("feature_set_names"),
                defaults.search.feature_set_names,
                allow_empty=False,
                dotted_key="research.defaults.search.feature_set_names",
            ),
            trainer_name=_resolve_strict_string(
                search_override.get("trainer_name"),
                defaults.search.trainer_name,
                dotted_key="research.defaults.search.trainer_name",
            ),
            preset_names=_resolve_string_list(
                search_override.get("preset_names"),
                defaults.search.preset_names,
                allow_empty=False,
                dotted_key="research.defaults.search.preset_names",
            ),
            working_target_id=_resolve_string(
                search_override.get("working_target_id"),
                defaults.search.working_target_id,
                dotted_key="research.defaults.search.working_target_id",
            ),
            max_worker_cap=_resolve_positive_int(
                search_override.get("max_worker_cap"),
                defaults.search.max_worker_cap,
                dotted_key="research.defaults.search.max_worker_cap",
            ),
            min_auto_workers=_resolve_positive_int(
                search_override.get("min_auto_workers"),
                defaults.search.min_auto_workers,
                dotted_key="research.defaults.search.min_auto_workers",
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
    _validate_worker_range(resolved.search)
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


def _resolve_strict_string(value: Any, default: str, *, dotted_key: str) -> str:
    if value is None:
        return str(default)
    if not isinstance(value, str):
        raise ValueError(f"{dotted_key} must be a non-empty string")
    candidate = value.strip()
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
    if exclusive_min and numeric <= 0.0:
        raise ValueError(f"{dotted_key} must be > 0")
    if not exclusive_min and numeric < 0.0:
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


def _validate_worker_range(search: ResearchSearchDefaults) -> None:
    if search.min_auto_workers > search.max_worker_cap:
        raise ValueError(
            "research.defaults.search.min_auto_workers must be <= research.defaults.search.max_worker_cap"
        )
