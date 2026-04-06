"""Feature inventory and named feature-set definitions for Stage 3 studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .schemas import FeatureInventoryRow


RAW_MARKET_COLUMNS = {
    "timestamp",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj close",
}

GROUP_DISPLAY_NAMES = {
    "momentum_trend": "Momentum / Trend",
    "volatility_range": "Volatility / Range",
    "volume_participation": "Volume / Participation",
    "price_structure": "Price Structure / Support-Resistance",
    "time_context": "Time / Context",
    "lag_features": "Lag Features",
    "statistical_rolling": "Statistical / Rolling",
    "excluded": "Excluded",
}

FEATURE_SET_DESCRIPTIONS = {
    "baseline_core": "Small mixed starter set: a few trend, volatility, context, and participation features for a balanced sanity-check baseline.",
    "momentum": "Trend and oscillator indicators such as moving averages, RSI, MACD, and similar directional features.",
    "volatility": "Range and volatility features such as ATR, Bollinger Band width/position, and related movement-intensity signals.",
    "context": "Market context features: time-of-day, participation/volume, and simple price-structure levels.",
    "lag_statistical": "Lagged values plus rolling statistics, useful for testing memory and distribution-shape effects.",
    "all_eligible": "All research-eligible features from the prepared matrix, excluding raw OHLCV and future/target columns.",
}


@dataclass(frozen=True)
class FeatureSetDefinition:
    """Named feature set used by a research study."""

    name: str
    display_name: str
    columns: List[str] = field(default_factory=list)
    group_names: List[str] = field(default_factory=list)
    description: str = ""


def group_features_by_prefix(columns: Iterable[str]) -> Dict[str, List[str]]:
    """Group feature columns by prefix before the first underscore."""
    grouped: Dict[str, List[str]] = {}
    for column in columns:
        name = str(column)
        prefix = name.split("_", 1)[0].lower()
        grouped.setdefault(prefix, []).append(name)
    return grouped


def is_raw_market_column(column: str) -> bool:
    """Return whether a column is a raw market field rather than a research feature."""
    return str(column).strip().lower() in RAW_MARKET_COLUMNS


def is_target_or_future_column(column: str) -> bool:
    """Return whether a column belongs to current/future target generation."""
    name = str(column)
    return name.startswith("Future_")


def classify_feature_group(column: str) -> str:
    """Map one prepared column to a stable Stage 3 research feature family."""
    name = str(column)
    lower_name = name.lower()

    if is_raw_market_column(name) or is_target_or_future_column(name):
        return "excluded"
    if "_lag_" in lower_name:
        return "lag_features"
    if any(token in lower_name for token in ("hour", "dayofweek", "dayofmonth", "month", "quarter", "is_")):
        return "time_context"
    if any(token in lower_name for token in ("volume", "obv")):
        return "volume_participation"
    if any(token in lower_name for token in ("pivot", "r1", "s1", "support", "resistance", "close_position", "price_change")):
        return "price_structure"
    if any(token in lower_name for token in ("atr", "bb_", "high_low_ratio", "range", "volatility")):
        return "volatility_range"
    if any(
        token in lower_name
        for token in (
            "returns_",
            "price_mean",
            "price_std",
            "price_min",
            "price_max",
            "z_score",
            "percentile",
            "skew",
            "kurt",
        )
    ):
        return "statistical_rolling"
    if any(token in lower_name for token in ("sma", "ema", "rsi", "macd", "stoch", "williams", "adx", "sar")):
        return "momentum_trend"
    return "statistical_rolling"


def build_feature_inventory(columns: Sequence[str]) -> List[FeatureInventoryRow]:
    """Build a deterministic feature inventory from the prepared feature matrix."""
    inventory_rows: List[FeatureInventoryRow] = []
    for name in sorted(str(column) for column in columns):
        eligible = not is_raw_market_column(name) and not is_target_or_future_column(name)
        exclusion_reason = ""
        if is_raw_market_column(name):
            exclusion_reason = "raw_market_column"
        elif is_target_or_future_column(name):
            exclusion_reason = "future_or_target_column"
        inventory_rows.append(
            FeatureInventoryRow(
                column=name,
                group=classify_feature_group(name),
                eligible=eligible,
                exclusion_reason=exclusion_reason,
            )
        )
    return inventory_rows


def build_named_feature_sets(columns: Sequence[str]) -> Dict[str, FeatureSetDefinition]:
    """Create deterministic Stage 3 feature sets from the current prepared matrix."""
    inventory_rows = build_feature_inventory(columns)
    eligible_by_group: Dict[str, List[str]] = {}
    for row in inventory_rows:
        if row.eligible:
            eligible_by_group.setdefault(row.group, []).append(row.column)

    for group_name in eligible_by_group:
        eligible_by_group[group_name] = sorted(eligible_by_group[group_name])

    momentum = list(eligible_by_group.get("momentum_trend", []))
    volatility = list(eligible_by_group.get("volatility_range", []))
    volume = list(eligible_by_group.get("volume_participation", []))
    price_structure = list(eligible_by_group.get("price_structure", []))
    context = list(eligible_by_group.get("time_context", []))
    lag_features = list(eligible_by_group.get("lag_features", []))
    statistical = list(eligible_by_group.get("statistical_rolling", []))

    baseline_core = _dedupe(
        momentum[:6]
        + volatility[:4]
        + volume[:2]
        + price_structure[:2]
        + context[:2]
    )
    all_eligible = _dedupe(
        momentum + volatility + volume + price_structure + context + lag_features + statistical
    )

    return {
        "baseline_core": FeatureSetDefinition(
            name="baseline_core",
            display_name="Baseline Core",
            columns=baseline_core,
            group_names=["momentum_trend", "volatility_range", "volume_participation", "price_structure", "time_context"],
            description="Small diversified baseline set pulled from the current prepared matrix.",
        ),
        "momentum": FeatureSetDefinition(
            name="momentum",
            display_name="Momentum",
            columns=momentum,
            group_names=["momentum_trend"],
            description="Trend and oscillator-style indicators.",
        ),
        "volatility": FeatureSetDefinition(
            name="volatility",
            display_name="Volatility",
            columns=volatility,
            group_names=["volatility_range"],
            description="Volatility and range features.",
        ),
        "context": FeatureSetDefinition(
            name="context",
            display_name="Context",
            columns=_dedupe(context + volume + price_structure),
            group_names=["time_context", "volume_participation", "price_structure"],
            description="Time, participation, and price-structure context features.",
        ),
        "lag_statistical": FeatureSetDefinition(
            name="lag_statistical",
            display_name="Lag / Statistical",
            columns=_dedupe(lag_features + statistical),
            group_names=["lag_features", "statistical_rolling"],
            description="Lagged values and rolling-statistical features.",
        ),
        "all_eligible": FeatureSetDefinition(
            name="all_eligible",
            display_name="All Eligible",
            columns=all_eligible,
            group_names=[
                "momentum_trend",
                "volatility_range",
                "volume_participation",
                "price_structure",
                "time_context",
                "lag_features",
                "statistical_rolling",
            ],
            description="Every eligible research feature from the prepared matrix.",
        ),
    }


def resolve_feature_sets(columns: Sequence[str], feature_set_names: Sequence[str] | None = None) -> List[FeatureSetDefinition]:
    """Resolve requested named feature sets against the current prepared matrix."""
    named_sets = build_named_feature_sets(columns)
    selected_names = list(feature_set_names or named_sets.keys())
    resolved_sets: List[FeatureSetDefinition] = []
    for name in selected_names:
        if name not in named_sets:
            raise ValueError(f"Unsupported Stage 3 feature set: {name}")
        resolved_sets.append(named_sets[name])
    return resolved_sets


def enrich_inventory_with_named_sets(
    inventory_rows: Sequence[FeatureInventoryRow],
    feature_sets: Sequence[FeatureSetDefinition],
) -> List[FeatureInventoryRow]:
    """Attach named-set membership information to feature inventory rows."""
    membership_map: Dict[str, List[str]] = {}
    for feature_set in feature_sets:
        for column in feature_set.columns:
            membership_map.setdefault(column, []).append(feature_set.name)

    enriched_rows: List[FeatureInventoryRow] = []
    for row in inventory_rows:
        enriched_rows.append(
            FeatureInventoryRow(
                column=row.column,
                group=row.group,
                eligible=row.eligible,
                included_in_named_sets=sorted(membership_map.get(row.column, [])),
                exclusion_reason=row.exclusion_reason,
                source=row.source,
            )
        )
    return enriched_rows


def get_group_display_name(group_name: str) -> str:
    """Return a user-friendly label for a Stage 3 feature group."""
    return GROUP_DISPLAY_NAMES.get(group_name, group_name.replace("_", " ").title())


def get_feature_set_display_name(feature_set_name: str) -> str:
    """Return a user-friendly label for a named research feature set."""
    named_sets = build_named_feature_sets([])
    feature_set = named_sets.get(feature_set_name)
    if feature_set is not None:
        return feature_set.display_name
    return feature_set_name.replace("_", " ").title()


def get_feature_set_description(feature_set_name: str) -> str:
    """Return a plain-language description for a named research feature set."""
    return FEATURE_SET_DESCRIPTIONS.get(feature_set_name, "")


def _dedupe(columns: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered
