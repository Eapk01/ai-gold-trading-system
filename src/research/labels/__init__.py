"""Target/label construction helpers for research experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class LegacyRuntimeDirectionSpec:
    """Definition of the current runtime target used in the legacy pipeline."""

    spec_id: str = "legacy_runtime_direction_1"
    display_name: str = "Legacy Runtime Direction (1 bar)"
    horizon_bars: int = 1
    price_column: str = "Close"


@dataclass(frozen=True)
class FixedHorizonDirectionSpec:
    """Definition of a fixed-horizon directional label."""

    spec_id: str
    display_name: str
    horizon_bars: int
    neutral_tolerance: float = 0.0
    price_column: str = "Close"


@dataclass(frozen=True)
class ReturnThresholdLabelSpec:
    """Definition of a percentage-return threshold label."""

    spec_id: str
    display_name: str
    horizon_bars: int
    return_threshold_pct: float
    price_column: str = "Close"


@dataclass(frozen=True)
class VolatilityAdjustedMoveSpec:
    """Definition of a volatility-adjusted future-move label."""

    spec_id: str
    display_name: str
    horizon_bars: int
    volatility_window: int
    volatility_multiplier: float
    price_column: str = "Close"


@dataclass(frozen=True)
class NeutralBandLabelSpec:
    """Definition of a neutral-band binary label with no-trade rows."""

    spec_id: str
    display_name: str
    horizon_bars: int
    neutral_band_pct: float
    price_column: str = "Close"


BinaryTargetSpec = (
    LegacyRuntimeDirectionSpec
    | FixedHorizonDirectionSpec
    | ReturnThresholdLabelSpec
    | VolatilityAdjustedMoveSpec
    | NeutralBandLabelSpec
)


def get_target_horizon_bars(spec: BinaryTargetSpec) -> int:
    """Return the lookahead horizon used by one target spec."""
    horizon_bars = int(getattr(spec, "horizon_bars", 0) or 0)
    if horizon_bars <= 0:
        raise ValueError("Target horizon must be a positive integer")
    return horizon_bars


def _build_future_return_pct(price_series: pd.Series, horizon_bars: int) -> pd.Series:
    """Return the explicit forward return from `t` to `t + horizon`."""
    future_price = price_series.shift(-int(horizon_bars))
    return ((future_price / price_series) - 1.0) * 100.0


def build_observed_persistence_labels(data: pd.DataFrame, spec: BinaryTargetSpec) -> pd.Series:
    """Build a past-observed analogue of a target for persistence-style baselines.

    Unlike `target.shift(1)`, this uses only information available at the prediction
    timestamp. For multi-bar future labels, it mirrors the target logic over the past
    observed horizon instead of reusing a previously computed future-looking label.
    """
    if getattr(spec, "price_column", "Close") not in data.columns:
        raise ValueError(f"Missing required price column: {getattr(spec, 'price_column', 'Close')}")

    price_column = getattr(spec, "price_column", "Close")
    price_series = pd.to_numeric(data[price_column], errors="coerce")
    horizon_bars = get_target_horizon_bars(spec)

    if isinstance(spec, LegacyRuntimeDirectionSpec):
        past_return_pct = (price_series.pct_change(periods=horizon_bars, fill_method=None)) * 100.0
        labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
        labels[past_return_pct > 0] = 1.0
        labels[past_return_pct <= 0] = 0.0
        labels[past_return_pct.isna()] = pd.NA
        return labels

    if isinstance(spec, FixedHorizonDirectionSpec):
        past_move = price_series - price_series.shift(horizon_bars)
        labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
        labels[past_move > float(spec.neutral_tolerance)] = 1.0
        labels[past_move < -float(spec.neutral_tolerance)] = 0.0
        return labels

    if isinstance(spec, ReturnThresholdLabelSpec):
        past_return_pct = ((price_series / price_series.shift(horizon_bars)) - 1.0) * 100.0
        labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
        labels[past_return_pct > float(spec.return_threshold_pct)] = 1.0
        labels[past_return_pct < -float(spec.return_threshold_pct)] = 0.0
        return labels

    if isinstance(spec, NeutralBandLabelSpec):
        return build_observed_persistence_labels(
            data,
            ReturnThresholdLabelSpec(
                spec_id=spec.spec_id,
                display_name=spec.display_name,
                horizon_bars=spec.horizon_bars,
                return_threshold_pct=spec.neutral_band_pct,
                price_column=spec.price_column,
            ),
        )

    if isinstance(spec, VolatilityAdjustedMoveSpec):
        returns_pct = price_series.pct_change(fill_method=None) * 100.0
        rolling_volatility = returns_pct.rolling(int(spec.volatility_window)).std()
        past_return_pct = ((price_series / price_series.shift(horizon_bars)) - 1.0) * 100.0
        dynamic_threshold = rolling_volatility * float(spec.volatility_multiplier)
        labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
        labels[past_return_pct > dynamic_threshold] = 1.0
        labels[past_return_pct < -dynamic_threshold] = 0.0
        return labels

    raise TypeError(f"Unsupported target spec: {type(spec)!r}")


def build_legacy_runtime_direction_labels(data: pd.DataFrame, spec: LegacyRuntimeDirectionSpec) -> pd.Series:
    """Reproduce the current runtime target definition exactly."""
    if spec.price_column not in data.columns:
        raise ValueError(f"Missing required price column: {spec.price_column}")

    close = pd.to_numeric(data[spec.price_column], errors="coerce")
    future_return_pct = _build_future_return_pct(close, get_target_horizon_bars(spec))
    labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
    labels[future_return_pct > 0] = 1.0
    labels[future_return_pct <= 0] = 0.0
    labels[future_return_pct.isna()] = pd.NA
    return labels


def build_fixed_horizon_direction_labels(data: pd.DataFrame, spec: FixedHorizonDirectionSpec) -> pd.Series:
    """Create a simple future-direction label from a price column.

    Values are:
    - `1` when the future move is above the tolerance
    - `0` when the future move is below the negative tolerance
    - `pd.NA` for neutral/unknown rows
    """
    if spec.price_column not in data.columns:
        raise ValueError(f"Missing required price column: {spec.price_column}")

    price_series = pd.to_numeric(data[spec.price_column], errors="coerce")
    future_price = price_series.shift(-get_target_horizon_bars(spec))
    future_return = future_price - price_series

    labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
    labels[future_return > float(spec.neutral_tolerance)] = 1.0
    labels[future_return < -float(spec.neutral_tolerance)] = 0.0
    return labels


def build_return_threshold_labels(data: pd.DataFrame, spec: ReturnThresholdLabelSpec) -> pd.Series:
    """Create a label from a future percentage return threshold."""
    if spec.price_column not in data.columns:
        raise ValueError(f"Missing required price column: {spec.price_column}")

    price_series = pd.to_numeric(data[spec.price_column], errors="coerce")
    future_return_pct = _build_future_return_pct(price_series, get_target_horizon_bars(spec))

    labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
    labels[future_return_pct > float(spec.return_threshold_pct)] = 1.0
    labels[future_return_pct < -float(spec.return_threshold_pct)] = 0.0
    return labels


def build_volatility_adjusted_move_labels(data: pd.DataFrame, spec: VolatilityAdjustedMoveSpec) -> pd.Series:
    """Create a label using future return relative to rolling volatility."""
    if spec.price_column not in data.columns:
        raise ValueError(f"Missing required price column: {spec.price_column}")

    price_series = pd.to_numeric(data[spec.price_column], errors="coerce")
    returns_pct = price_series.pct_change(fill_method=None) * 100.0
    rolling_volatility = returns_pct.rolling(int(spec.volatility_window)).std()
    future_return_pct = _build_future_return_pct(price_series, get_target_horizon_bars(spec))
    dynamic_threshold = rolling_volatility * float(spec.volatility_multiplier)

    labels = pd.Series(pd.NA, index=data.index, dtype="Float64")
    labels[future_return_pct > dynamic_threshold] = 1.0
    labels[future_return_pct < -dynamic_threshold] = 0.0
    return labels


def build_neutral_band_labels(data: pd.DataFrame, spec: NeutralBandLabelSpec) -> pd.Series:
    """Create a binary label with a no-trade neutral band around zero return."""
    return build_return_threshold_labels(
        data,
        ReturnThresholdLabelSpec(
            spec_id=spec.spec_id,
            display_name=spec.display_name,
            horizon_bars=spec.horizon_bars,
            return_threshold_pct=spec.neutral_band_pct,
            price_column=spec.price_column,
        ),
    )


def build_binary_target_labels(data: pd.DataFrame, spec: BinaryTargetSpec) -> pd.Series:
    """Dispatch a target spec to the appropriate binary label builder."""
    if isinstance(spec, LegacyRuntimeDirectionSpec):
        return build_legacy_runtime_direction_labels(data, spec)
    if isinstance(spec, FixedHorizonDirectionSpec):
        return build_fixed_horizon_direction_labels(data, spec)
    if isinstance(spec, ReturnThresholdLabelSpec):
        return build_return_threshold_labels(data, spec)
    if isinstance(spec, VolatilityAdjustedMoveSpec):
        return build_volatility_adjusted_move_labels(data, spec)
    if isinstance(spec, NeutralBandLabelSpec):
        return build_neutral_band_labels(data, spec)
    raise TypeError(f"Unsupported target spec: {type(spec)!r}")


def serialize_target_spec(spec: BinaryTargetSpec) -> dict[str, Any]:
    """Return a JSON-safe representation of a binary target spec."""
    payload = asdict(spec)
    payload["target_type"] = spec.__class__.__name__
    return payload


def spec_display_name(spec: BinaryTargetSpec) -> str:
    """Return a stable display name for a target spec."""
    return getattr(spec, "display_name", "") or getattr(spec, "spec_id", spec.__class__.__name__)


def spec_target_type(spec: BinaryTargetSpec) -> str:
    """Return a stable target type identifier."""
    return spec.__class__.__name__
