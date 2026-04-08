"""Bounded Stage 5 preset definitions for automated search."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


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

STAGE5_LSTM_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "lookback_window": 20,
        "hidden_size": 24,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "class_weighting": True,
        "early_stopping_patience": 4,
        "decision_threshold": 0.5,
        "seed": 42,
    },
    "balanced": {
        "lookback_window": 32,
        "hidden_size": 48,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
        "class_weighting": True,
        "early_stopping_patience": 5,
        "decision_threshold": 0.5,
        "seed": 42,
    },
    "capacity": {
        "lookback_window": 48,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 40,
        "class_weighting": True,
        "early_stopping_patience": 6,
        "decision_threshold": 0.5,
        "seed": 42,
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
    "capacity": "Larger, less constrained models. This preset gives the model more expressive power but carries higher overfitting risk.",
}


def list_stage5_preset_names() -> List[str]:
    """Return the canonical Stage 5 preset ids."""
    return list(STAGE5_PRESET_DISPLAY_NAMES.keys())


def resolve_stage5_preset_definitions(
    trainer_name: str | Iterable[str],
    model_names: Iterable[str],
    preset_names: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return Stage 5 preset definitions filtered to the configured trainer."""
    if preset_names is None:
        preset_names = model_names
        model_names = trainer_name  # type: ignore[assignment]
        trainer_name = "current_ensemble"

    normalized_trainer = str(trainer_name or "current_ensemble").strip().lower()
    requested_presets = [str(name).strip() for name in preset_names]
    unknown_presets = [preset_name for preset_name in requested_presets if preset_name not in STAGE5_PRESET_DISPLAY_NAMES]
    if unknown_presets:
        raise ValueError(f"Unknown Stage 5 preset names: {', '.join(unknown_presets)}")

    if normalized_trainer == "current_ensemble":
        configured_models = [str(name).strip() for name in model_names]
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

        return {
            preset_name: {
                model_name: dict(STAGE5_CURRENT_ENSEMBLE_PRESETS[preset_name][model_name])
                for model_name in configured_models
            }
            for preset_name in requested_presets
        }

    if normalized_trainer == "lstm":
        return {
            preset_name: dict(STAGE5_LSTM_PRESETS[preset_name])
            for preset_name in requested_presets
        }

    raise ValueError(f"Unsupported trainer for Stage 5 presets: {trainer_name}")


def get_stage5_preset_display_name(preset_name: str) -> str:
    """Return a user-facing label for a Stage 5 search preset."""
    return STAGE5_PRESET_DISPLAY_NAMES.get(preset_name, preset_name.replace("_", " ").title())


def get_stage5_preset_description(preset_name: str) -> str:
    """Return a plain-language description for a Stage 5 search preset."""
    return STAGE5_PRESET_DESCRIPTIONS.get(preset_name, "")


def build_stage5_preset_catalog(
    trainer_name: str,
    model_names: Iterable[str],
    selected_preset_names: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Return GUI-ready preset rows for one trainer."""
    selected = {str(name).strip() for name in (selected_preset_names or [])}
    normalized_trainer = str(trainer_name or "current_ensemble").strip().lower()
    preset_names = list_stage5_preset_names()
    preset_definitions = resolve_stage5_preset_definitions(
        normalized_trainer,
        model_names,
        preset_names,
    )

    rows: List[Dict[str, Any]] = []
    for preset_name in preset_names:
        preset_payload = dict(preset_definitions.get(preset_name) or {})
        if normalized_trainer == "current_ensemble":
            summary = ", ".join(sorted(preset_payload.keys())) if preset_payload else "Unavailable"
        else:
            summary = f"{len(preset_payload)} parameters" if preset_payload else "Unavailable"
        rows.append(
            {
                "id": preset_name,
                "display_name": get_stage5_preset_display_name(preset_name),
                "description": get_stage5_preset_description(preset_name),
                "selected": preset_name in selected,
                "trainer_name": normalized_trainer,
                "summary": summary,
            }
        )
    return rows
