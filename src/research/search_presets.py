"""Bounded Stage 5 preset definitions for automated search."""

from __future__ import annotations

from typing import Any, Dict, Iterable


STAGE5_CURRENT_ENSEMBLE_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "conservative": {
        "logistic_regression": {
            "C": 0.25,
            "max_iter": 2000,
            "class_weight": "balanced",
        },
        "random_forest": {
            "n_estimators": 150,
            "max_depth": 4,
            "min_samples_leaf": 10,
            "random_state": 42,
        },
        "xgboost": {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        },
    },
    "balanced": {
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
        },
        "random_forest": {
            "n_estimators": 250,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "xgboost": {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "eval_metric": "logloss",
        },
    },
    "capacity": {
        "logistic_regression": {
            "C": 4.0,
            "max_iter": 2000,
            "class_weight": "balanced",
        },
        "random_forest": {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
        },
    },
}

STAGE5_PRESET_DISPLAY_NAMES = {
    "conservative": "Conservative",
    "balanced": "Balanced",
    "capacity": "Capacity",
}

STAGE5_PRESET_DESCRIPTIONS = {
    "conservative": "Smaller, more regularized models. This is the safer, lower-capacity preset intended to reduce overfitting.",
    "balanced": "Middle-ground preset. Moderate capacity with moderate regularization for a general-purpose comparison point.",
    "capacity": "Larger, less constrained models. This preset gives the ensemble more expressive power but carries higher overfitting risk.",
}


def resolve_stage5_preset_definitions(
    model_names: Iterable[str],
    preset_names: Iterable[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return Stage 5 preset definitions filtered to the configured ensemble models."""
    configured_models = [str(name).strip() for name in model_names]
    requested_presets = [str(name).strip() for name in preset_names]
    unsupported_models = sorted(
        {
            model_name
            for model_name in configured_models
            if any(model_name not in STAGE5_CURRENT_ENSEMBLE_PRESETS.get(preset_name, {}) for preset_name in requested_presets)
        }
    )
    if unsupported_models:
        raise ValueError(
            "Stage 5 search is missing preset mappings for configured ensemble models: "
            + ", ".join(unsupported_models)
        )

    unknown_presets = [preset_name for preset_name in requested_presets if preset_name not in STAGE5_CURRENT_ENSEMBLE_PRESETS]
    if unknown_presets:
        raise ValueError(f"Unknown Stage 5 preset names: {', '.join(unknown_presets)}")

    return {
        preset_name: {
            model_name: dict(STAGE5_CURRENT_ENSEMBLE_PRESETS[preset_name][model_name])
            for model_name in configured_models
        }
        for preset_name in requested_presets
    }


def get_stage5_preset_display_name(preset_name: str) -> str:
    """Return a user-facing label for a Stage 5 search preset."""
    return STAGE5_PRESET_DISPLAY_NAMES.get(preset_name, preset_name.replace("_", " ").title())


def get_stage5_preset_description(preset_name: str) -> str:
    """Return a plain-language description for a Stage 5 search preset."""
    return STAGE5_PRESET_DESCRIPTIONS.get(preset_name, "")
