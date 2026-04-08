"""Stage 5 target catalog helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from ..labels import LegacyRuntimeDirectionSpec, VolatilityAdjustedMoveSpec, serialize_target_spec


STAGE5_TARGET_DESCRIPTIONS = {
    "return_threshold_h3_0_05pct": "Binary target based on a future 3-bar move greater than 0.05% in either direction.",
    "vol_adjusted_h3_x1_0": "Binary target based on a future 3-bar move relative to rolling volatility.",
}


def build_stage5_target_specs(
    *,
    runtime_target_column: str,
    working_target_spec: object,
) -> List[object]:
    """Return the bounded Stage 5 target spec list."""
    try:
        legacy_horizon = int(str(runtime_target_column).rsplit("_", 1)[-1])
    except ValueError:
        legacy_horizon = 1

    return [
        working_target_spec,
        LegacyRuntimeDirectionSpec(
            spec_id=f"legacy_{runtime_target_column.lower()}",
            display_name=f"Legacy Runtime Target ({runtime_target_column})",
            horizon_bars=legacy_horizon,
        ),
        VolatilityAdjustedMoveSpec(
            spec_id="vol_adjusted_h3_x1_0",
            display_name="Volatility Adjusted (3 bars, 1.0x vol)",
            horizon_bars=3,
            volatility_window=20,
            volatility_multiplier=1.0,
        ),
    ]


def describe_stage5_target(spec_id: str, display_name: str) -> str:
    """Return a plain-language description for a Stage 5 target."""
    if str(spec_id or "").startswith("legacy_"):
        return "Reuses the current runtime target column so search can compare against the legacy live-trading label."
    return STAGE5_TARGET_DESCRIPTIONS.get(spec_id, display_name)


def build_stage5_target_catalog_rows(
    *,
    target_specs: List[object],
    selected_target_ids: List[str],
) -> List[Dict[str, Any]]:
    """Build GUI-ready catalog rows for Stage 5 targets."""
    rows: List[Dict[str, Any]] = []
    for target_spec in target_specs:
        serialized = serialize_target_spec(target_spec)
        spec_id = str(serialized.get("spec_id") or "")
        display_name = str(serialized.get("display_name") or spec_id)
        rows.append(
            {
                "id": spec_id,
                "display_name": display_name,
                "description": describe_stage5_target(spec_id, display_name),
                "selected": spec_id in selected_target_ids,
            }
        )
    return rows
