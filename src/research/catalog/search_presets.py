"""Bounded preset definitions for the primary research search."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


SEARCH_CURRENT_ENSEMBLE_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
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

def _lstm_variant(
    variant_name: str,
    *,
    feature_mode: str,
    lookback_window: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    dense_hidden_size: int,
    dense_dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    bidirectional: bool = False,
    early_stopping_patience: int = 5,
) -> Dict[str, Any]:
    return {
        "preset_variant_name": variant_name,
        "architecture_name": "lstm_v2_dense_head",
        "feature_mode": feature_mode,
        "lookback_window": lookback_window,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "dense_hidden_size": dense_hidden_size,
        "dense_dropout": dense_dropout,
        "activation": "gelu",
        "bidirectional": bidirectional,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_weighting": True,
        "early_stopping_patience": early_stopping_patience,
        "decision_threshold": 0.5,
        "seed": 42,
    }


SEARCH_LSTM_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "variants": [
            _lstm_variant(
                "eng_l24_h32_d24",
                feature_mode="engineered",
                lookback_window=24,
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
                dense_hidden_size=24,
                dense_dropout=0.05,
                learning_rate=0.001,
                batch_size=32,
                epochs=20,
                weight_decay=0.0001,
                early_stopping_patience=4,
            ),
            _lstm_variant(
                "combo_l24_h32_d32",
                feature_mode="combined",
                lookback_window=24,
                hidden_size=32,
                num_layers=1,
                dropout=0.05,
                dense_hidden_size=32,
                dense_dropout=0.08,
                learning_rate=0.001,
                batch_size=32,
                epochs=22,
                weight_decay=0.0001,
                early_stopping_patience=4,
            ),
        ],
    },
    "balanced": {
        "variants": [
            _lstm_variant(
                "eng_l32_h48_d48",
                feature_mode="engineered",
                lookback_window=32,
                hidden_size=48,
                num_layers=2,
                dropout=0.1,
                dense_hidden_size=48,
                dense_dropout=0.1,
                learning_rate=0.001,
                batch_size=32,
                epochs=30,
                weight_decay=0.0001,
            ),
            _lstm_variant(
                "combo_l32_h48_d48",
                feature_mode="combined",
                lookback_window=32,
                hidden_size=48,
                num_layers=2,
                dropout=0.1,
                dense_hidden_size=48,
                dense_dropout=0.1,
                learning_rate=0.001,
                batch_size=32,
                epochs=30,
                weight_decay=0.0001,
            ),
            _lstm_variant(
                "raw_l32_h48_d32",
                feature_mode="raw_market",
                lookback_window=32,
                hidden_size=48,
                num_layers=1,
                dropout=0.05,
                dense_hidden_size=32,
                dense_dropout=0.1,
                learning_rate=0.001,
                batch_size=32,
                epochs=28,
                weight_decay=0.0001,
            ),
            _lstm_variant(
                "combo_l48_h64_d48",
                feature_mode="combined",
                lookback_window=48,
                hidden_size=64,
                num_layers=2,
                dropout=0.15,
                dense_hidden_size=48,
                dense_dropout=0.12,
                learning_rate=0.0007,
                batch_size=64,
                epochs=34,
                weight_decay=0.00015,
            ),
            _lstm_variant(
                "eng_l24_h48_d32",
                feature_mode="engineered",
                lookback_window=24,
                hidden_size=48,
                num_layers=1,
                dropout=0.05,
                dense_hidden_size=32,
                dense_dropout=0.08,
                learning_rate=0.001,
                batch_size=32,
                epochs=28,
                weight_decay=0.0001,
            ),
            _lstm_variant(
                "raw_l48_h64_d48",
                feature_mode="raw_market",
                lookback_window=48,
                hidden_size=64,
                num_layers=2,
                dropout=0.15,
                dense_hidden_size=48,
                dense_dropout=0.12,
                learning_rate=0.0007,
                batch_size=64,
                epochs=34,
                weight_decay=0.00015,
            ),
        ],
    },
    "capacity": {
        "variants": [
            _lstm_variant(
                "combo_l48_h64_d64",
                feature_mode="combined",
                lookback_window=48,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                dense_hidden_size=64,
                dense_dropout=0.15,
                learning_rate=0.0005,
                batch_size=64,
                epochs=40,
                weight_decay=0.0002,
                early_stopping_patience=6,
            ),
            _lstm_variant(
                "raw_l48_h64_d64",
                feature_mode="raw_market",
                lookback_window=48,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                dense_hidden_size=64,
                dense_dropout=0.15,
                learning_rate=0.0005,
                batch_size=64,
                epochs=40,
                weight_decay=0.0002,
                early_stopping_patience=6,
            ),
            _lstm_variant(
                "combo_l64_h80_d64",
                feature_mode="combined",
                lookback_window=64,
                hidden_size=80,
                num_layers=2,
                dropout=0.22,
                dense_hidden_size=64,
                dense_dropout=0.18,
                learning_rate=0.0005,
                batch_size=64,
                epochs=44,
                weight_decay=0.00025,
                early_stopping_patience=6,
            ),
            _lstm_variant(
                "raw_l64_h80_d64",
                feature_mode="raw_market",
                lookback_window=64,
                hidden_size=80,
                num_layers=2,
                dropout=0.22,
                dense_hidden_size=64,
                dense_dropout=0.18,
                learning_rate=0.0005,
                batch_size=64,
                epochs=44,
                weight_decay=0.00025,
                early_stopping_patience=6,
            ),
            _lstm_variant(
                "combo_l48_h64_bidir",
                feature_mode="combined",
                lookback_window=48,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                dense_hidden_size=80,
                dense_dropout=0.18,
                learning_rate=0.0005,
                batch_size=64,
                epochs=42,
                weight_decay=0.00025,
                bidirectional=True,
                early_stopping_patience=6,
            ),
            _lstm_variant(
                "raw_l48_h64_bidir",
                feature_mode="raw_market",
                lookback_window=48,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                dense_hidden_size=80,
                dense_dropout=0.18,
                learning_rate=0.0005,
                batch_size=64,
                epochs=42,
                weight_decay=0.00025,
                bidirectional=True,
                early_stopping_patience=6,
            ),
        ],
    },
}

_LSTM_PRESET_BASE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "architecture_name": "lstm_v2_dense_head",
        "feature_mode": "engineered",
        "lookback_window": 24,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "dense_hidden_size": 24,
        "dense_dropout": 0.05,
        "activation": "gelu",
        "bidirectional": False,
        "weight_decay": 0.0001,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "class_weighting": True,
        "early_stopping_patience": 4,
        "decision_threshold": 0.5,
        "seed": 42,
    },
    "balanced": {
        "architecture_name": "lstm_v2_dense_head",
        "feature_mode": "combined",
        "lookback_window": 32,
        "hidden_size": 48,
        "num_layers": 2,
        "dropout": 0.1,
        "dense_hidden_size": 48,
        "dense_dropout": 0.1,
        "activation": "gelu",
        "bidirectional": False,
        "weight_decay": 0.0001,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
        "class_weighting": True,
        "early_stopping_patience": 5,
        "decision_threshold": 0.5,
        "seed": 42,
    },
    "capacity": {
        "architecture_name": "lstm_v2_dense_head",
        "feature_mode": "combined",
        "lookback_window": 48,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "dense_hidden_size": 64,
        "dense_dropout": 0.15,
        "activation": "gelu",
        "bidirectional": False,
        "weight_decay": 0.0002,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 40,
        "class_weighting": True,
        "early_stopping_patience": 6,
        "decision_threshold": 0.5,
        "seed": 42,
    },
}

SEARCH_PRESET_DISPLAY_NAMES = {
    "conservative": "Conservative",
    "balanced": "Balanced",
    "capacity": "Capacity",
}

SEARCH_PRESET_DESCRIPTIONS = {
    "conservative": "Smaller, more regularized models. This is the safer, lower-capacity preset intended to reduce overfitting.",
    "balanced": "Middle-ground preset. Moderate capacity with moderate regularization for a general-purpose comparison point.",
    "capacity": "Larger, less constrained models. This preset gives the model more expressive power but carries higher overfitting risk.",
}


def list_search_preset_names() -> List[str]:
    """Return the canonical research search preset ids."""
    return list(SEARCH_PRESET_DISPLAY_NAMES.keys())


def resolve_search_preset_definitions(
    trainer_name: str | Iterable[str],
    model_names: Iterable[str],
    preset_names: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return preset definitions filtered to the configured trainer."""
    if preset_names is None:
        preset_names = model_names
        model_names = trainer_name  # type: ignore[assignment]
        trainer_name = "current_ensemble"

    normalized_trainer = str(trainer_name or "current_ensemble").strip().lower()
    requested_presets = [str(name).strip() for name in preset_names]
    unknown_presets = [preset_name for preset_name in requested_presets if preset_name not in SEARCH_PRESET_DISPLAY_NAMES]
    if unknown_presets:
        raise ValueError(f"Unknown research search preset names: {', '.join(unknown_presets)}")

    if normalized_trainer == "current_ensemble":
        configured_models = [str(name).strip() for name in model_names]
        unsupported_models = sorted(
            {
                model_name
                for model_name in configured_models
                if any(model_name not in SEARCH_CURRENT_ENSEMBLE_PRESETS.get(preset_name, {}) for preset_name in requested_presets)
            }
        )
        if unsupported_models:
            raise ValueError(
                "Research search is missing preset mappings for configured ensemble models: "
                + ", ".join(unsupported_models)
            )

        return {
            preset_name: {
                model_name: dict(SEARCH_CURRENT_ENSEMBLE_PRESETS[preset_name][model_name])
                for model_name in configured_models
            }
            for preset_name in requested_presets
        }

    if normalized_trainer == "lstm":
        return {
            preset_name: {
                **dict(_LSTM_PRESET_BASE_SETTINGS[preset_name]),
                "variants": [dict(variant) for variant in SEARCH_LSTM_PRESETS[preset_name]["variants"]],
            }
            for preset_name in requested_presets
        }

    raise ValueError(f"Unsupported trainer for research search presets: {trainer_name}")


def get_search_preset_display_name(preset_name: str) -> str:
    """Return a user-facing label for a research search preset."""
    return SEARCH_PRESET_DISPLAY_NAMES.get(preset_name, preset_name.replace("_", " ").title())


def get_search_preset_description(preset_name: str) -> str:
    """Return a plain-language description for a research search preset."""
    return SEARCH_PRESET_DESCRIPTIONS.get(preset_name, "")


def expand_lstm_search_preset_variants(preset_name: str, preset_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return curated LSTM variant parameter dictionaries for one preset."""
    variants = preset_payload.get("variants")
    if isinstance(variants, list) and variants:
        return [
            {
                **{key: value for key, value in preset_payload.items() if key != "variants"},
                **dict(variant),
                "preset_name": preset_name,
                "preset_variant_name": str(dict(variant).get("preset_variant_name") or f"{preset_name}_{index}"),
            }
            for index, variant in enumerate(variants, start=1)
        ]
    return [
        {
            **dict(preset_payload),
            "preset_name": preset_name,
            "preset_variant_name": str(preset_payload.get("preset_variant_name") or preset_name),
        }
    ]


def count_search_preset_variants(trainer_name: str, preset_payload: Dict[str, Any]) -> int:
    """Return how many candidate configs a preset expands into."""
    if str(trainer_name or "").strip().lower() != "lstm":
        return 1
    variants = preset_payload.get("variants")
    return len(variants) if isinstance(variants, list) and variants else 1


def describe_lstm_search_variants(preset_payload: Dict[str, Any]) -> str:
    variants = [dict(variant) for variant in preset_payload.get("variants", []) if isinstance(variant, dict)]
    if not variants:
        return (
            f"1 variant | {preset_payload.get('feature_mode', 'engineered')} | "
            f"lookback {preset_payload.get('lookback_window')}"
        )
    feature_modes = sorted({str(variant.get("feature_mode") or "engineered") for variant in variants})
    lookbacks = sorted({int(variant.get("lookback_window") or 0) for variant in variants if variant.get("lookback_window")})
    bidirectional_count = len([variant for variant in variants if bool(variant.get("bidirectional"))])
    mode_summary = " + ".join(feature_modes)
    if lookbacks:
        lookback_summary = f"lookback {lookbacks[0]}" if len(lookbacks) == 1 else f"lookback {lookbacks[0]}-{lookbacks[-1]}"
    else:
        lookback_summary = "lookback mixed"
    bidirectional_summary = f" | {bidirectional_count} bidirectional" if bidirectional_count else ""
    architecture = str(variants[0].get("architecture_name") or "lstm_v2_dense_head")
    return f"{architecture} | {len(variants)} variants | {mode_summary} | {lookback_summary}{bidirectional_summary}"


def build_search_preset_catalog(
    trainer_name: str,
    model_names: Iterable[str],
    selected_preset_names: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Return GUI-ready preset rows for one trainer."""
    selected = {str(name).strip() for name in (selected_preset_names or [])}
    normalized_trainer = str(trainer_name or "current_ensemble").strip().lower()
    preset_names = list_search_preset_names()
    preset_definitions = resolve_search_preset_definitions(
        normalized_trainer,
        model_names,
        preset_names,
    )

    rows: List[Dict[str, Any]] = []
    for preset_name in preset_names:
        preset_payload = dict(preset_definitions.get(preset_name) or {})
        if normalized_trainer == "current_ensemble":
            summary = ", ".join(sorted(preset_payload.keys())) if preset_payload else "Unavailable"
            variant_count = 1 if preset_payload else 0
        else:
            variant_count = count_search_preset_variants(normalized_trainer, preset_payload)
            summary = describe_lstm_search_variants(preset_payload) if preset_payload else "Unavailable"
        rows.append(
            {
                "id": preset_name,
                "display_name": get_search_preset_display_name(preset_name),
                "description": get_search_preset_description(preset_name),
                "selected": preset_name in selected,
                "trainer_name": normalized_trainer,
                "summary": summary,
                "variant_count": variant_count,
            }
        )
    return rows
