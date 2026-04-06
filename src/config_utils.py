"""
Configuration loading, normalization, and validation helpers.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_TARGET_COLUMN = "Future_Direction_1"


class ConfigValidationError(ValueError):
    """Raised when the application configuration is invalid."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries recursively without mutating inputs."""
    merged = deepcopy(base)

    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_default_config() -> Dict[str, Any]:
    """Return a baseline configuration used to fill optional values."""
    from src.research.defaults import get_builtin_research_defaults

    research_defaults = get_builtin_research_defaults(DEFAULT_TARGET_COLUMN).to_dict()
    return {
        "trading": {
            "symbol": "XAUUSD",
            "timeframe": "5m",
            "position_size": 0.01,
            "stop_loss_pips": 50,
            "take_profit_pips": 100,
            "confidence_threshold": 0.60,
        },
        "data_sources": {
            "primary": "local_csv",
            "dataset_directory": "data/imports",
            "min_rows": 100,
        },
        "ai_model": {
            "type": "ensemble",
            "models": ["random_forest", "xgboost", "logistic_regression"],
            "models_directory": "models",
            "retrain_interval": 168,
            "lookback_periods": 100,
            "target_column": DEFAULT_TARGET_COLUMN,
        },
        "features": {
            "technical_indicators": ["sma_5", "sma_20", "rsi_14", "macd", "atr_14"],
            "market_sentiment": ["vix", "dxy"],
        },
        "research": {
            "experiments_directory": "reports/experiments",
            "candidate_models_directory": "models/candidates",
            "defaults": {key: value for key, value in research_defaults.items() if key != "runtime_target_column"},
        },
        "risk_management": {
            "max_daily_loss": 30.0,
            "max_positions": 3,
            "risk_per_trade": 0.02,
            "drawdown_limit": 0.15,
        },
        "backtest": {
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 10000.0,
            "commission": 0.0001,
            "slippage": 0.0002,
            "signal_confidence_threshold": 0.60,
        },
        "database": {
            "type": "sqlite",
            "path": "data/trading_system.db",
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/trading_system.log",
            "max_file_size": "10MB",
            "backup_count": 5,
        },
        "app": {
            "startup": {
                "autoload_latest_model": True,
                "autoconnect_broker": True,
            }
        },
        "brokers": {
            "exness": {
                "enabled": False,
                "server": "",
                "login": "",
                "terminal_path": "",
                "timeout": 30,
                "max_retries": 3,
            },
            "profiles": {},
            "default_profile": "",
        },
        "live_trading": {
            "enabled_demo_only": True,
            "poll_interval_seconds": 5,
            "inactive_poll_interval_seconds": 30,
            "signal_confidence_threshold": 0.60,
            "startup_candle_buffer": 150,
            "stale_candle_multiplier": 4,
        },
    }


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load, normalize, and validate application configuration."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file_obj:
        raw_config = yaml.safe_load(file_obj) or {}

    config = _deep_merge(get_default_config(), raw_config)
    validate_config(config)
    return config


def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> None:
    """Validate and persist application configuration."""
    validate_config(config)
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=False)


def ensure_runtime_directories(config: Dict[str, Any]) -> List[str]:
    """Create directories required by the runtime and return their paths."""
    directories = {"data", "logs", "models", "reports"}

    db_path = config.get("database", {}).get("path", "")
    if db_path:
        db_parent = Path(db_path).parent
        if str(db_parent) not in ("", "."):
            directories.add(str(db_parent))

    log_path = config.get("logging", {}).get("file_path", "")
    if log_path:
        log_parent = Path(log_path).parent
        if str(log_parent) not in ("", "."):
            directories.add(str(log_parent))

    dataset_directory = config.get("data_sources", {}).get("dataset_directory", "")
    if dataset_directory:
        directories.add(str(dataset_directory))

    models_directory = config.get("ai_model", {}).get("models_directory", "")
    if models_directory:
        directories.add(str(models_directory))

    experiments_directory = config.get("research", {}).get("experiments_directory", "")
    if experiments_directory:
        directories.add(str(experiments_directory))

    candidate_models_directory = config.get("research", {}).get("candidate_models_directory", "")
    if candidate_models_directory:
        directories.add(str(candidate_models_directory))

    created = []
    for directory in sorted(directories):
        Path(directory).mkdir(parents=True, exist_ok=True)
        created.append(directory)

    return created


def get_target_column(config: Dict[str, Any]) -> str:
    """Return the configured model target column with a stable fallback."""
    raw_value = config.get("ai_model", {}).get("target_column", DEFAULT_TARGET_COLUMN)
    target_column = str(raw_value).strip()
    return target_column or DEFAULT_TARGET_COLUMN


def get_effective_confidence_threshold(config: Dict[str, Any], mode: str) -> float:
    """Return the effective confidence threshold for a runtime mode.

    `trading.confidence_threshold` is treated as the shared default.
    `live_trading.signal_confidence_threshold` and
    `backtest.signal_confidence_threshold` are mode-specific overrides.
    """
    trading_default = float(config.get("trading", {}).get("confidence_threshold", 0.60))

    if mode == "live_trading":
        return float(config.get("live_trading", {}).get("signal_confidence_threshold", trading_default))
    if mode == "backtest":
        return float(config.get("backtest", {}).get("signal_confidence_threshold", trading_default))

    raise ValueError(f"Unsupported confidence-threshold mode: {mode}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration and raise ConfigValidationError on issues."""
    errors: List[str] = []

    _require_keys(
        config,
        {
            "trading": ["symbol", "timeframe", "position_size", "stop_loss_pips", "take_profit_pips"],
            "data_sources": ["primary", "dataset_directory", "min_rows"],
            "ai_model": ["type", "models", "lookback_periods"],
            "risk_management": ["max_daily_loss", "max_positions", "risk_per_trade", "drawdown_limit"],
            "backtest": ["initial_capital", "commission", "slippage"],
            "database": ["path"],
            "logging": ["level", "file_path"],
        },
        errors,
    )

    _validate_numeric(config, "trading.position_size", min_value=0, errors=errors)
    _validate_numeric(config, "trading.confidence_threshold", min_value=0, max_value=1, errors=errors)
    _validate_numeric(config, "trading.stop_loss_pips", min_value=0, errors=errors)
    _validate_numeric(config, "trading.take_profit_pips", min_value=0, errors=errors)
    _validate_numeric(config, "data_sources.min_rows", min_value=1, errors=errors, integer=True)
    _validate_numeric(config, "ai_model.lookback_periods", min_value=1, errors=errors, integer=True)
    _validate_numeric(config, "live_trading.poll_interval_seconds", min_value=1, errors=errors, integer=True)
    _validate_numeric(config, "live_trading.inactive_poll_interval_seconds", min_value=1, errors=errors, integer=True)
    _validate_numeric(config, "live_trading.signal_confidence_threshold", min_value=0, max_value=1, errors=errors)
    _validate_numeric(config, "live_trading.startup_candle_buffer", min_value=10, errors=errors, integer=True)
    _validate_numeric(config, "live_trading.stale_candle_multiplier", min_value=1, errors=errors, integer=True)
    _validate_numeric(config, "risk_management.max_daily_loss", min_value=0, errors=errors)
    _validate_numeric(config, "risk_management.max_positions", min_value=1, errors=errors, integer=True)
    _validate_numeric(config, "risk_management.risk_per_trade", min_value=0, max_value=1, errors=errors)
    _validate_numeric(config, "risk_management.drawdown_limit", min_value=0, max_value=1, errors=errors)
    _validate_numeric(config, "backtest.initial_capital", min_value=0, errors=errors)
    _validate_numeric(config, "backtest.commission", min_value=0, errors=errors)
    _validate_numeric(config, "backtest.slippage", min_value=0, errors=errors)
    _validate_numeric(config, "backtest.signal_confidence_threshold", min_value=0, max_value=1, errors=errors)
    models = config.get("ai_model", {}).get("models", [])
    if not isinstance(models, list) or not models:
        errors.append("ai_model.models must be a non-empty list")

    target_column = get_target_column(config)
    if not target_column:
        errors.append("ai_model.target_column must be a non-empty string")

    supported_data_sources = {"local_csv"}
    primary_source = str(config.get("data_sources", {}).get("primary", "")).lower()
    if primary_source not in supported_data_sources:
        errors.append(
            f"data_sources.primary must be one of {sorted(supported_data_sources)}, got '{primary_source}'"
        )

    if config.get("brokers", {}).get("exness", {}).get("enabled"):
        _validate_exness_settings(config["brokers"]["exness"], errors)

    try:
        from src.research.defaults import resolve_research_defaults

        resolve_research_defaults(config)
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        raise ConfigValidationError("Configuration validation failed:\n- " + "\n- ".join(errors))


def _validate_exness_settings(exness_config: Dict[str, Any], errors: List[str]) -> None:
    required_fields = ["server", "login"]
    missing = [field for field in required_fields if not str(exness_config.get(field, "")).strip()]
    if missing:
        errors.append(f"brokers.exness is missing required fields: {', '.join(missing)}")

    if find_spec("MetaTrader5") is None:
        errors.append("MetaTrader5 package is required for Exness broker support but is not installed")


def _require_keys(config: Dict[str, Any], required_map: Dict[str, List[str]], errors: List[str]) -> None:
    for section, keys in required_map.items():
        section_value = config.get(section)
        if not isinstance(section_value, dict):
            errors.append(f"{section} must be a mapping")
            continue

        for key in keys:
            if key not in section_value:
                errors.append(f"Missing required config key: {section}.{key}")


def _validate_numeric(
    config: Dict[str, Any],
    dotted_key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    errors: List[str],
    integer: bool = False,
) -> None:
    found, value = _resolve(config, dotted_key)
    if not found:
        return

    if integer:
        if not isinstance(value, int):
            errors.append(f"{dotted_key} must be an integer")
            return
    else:
        if not isinstance(value, (int, float)):
            errors.append(f"{dotted_key} must be numeric")
            return

    if min_value is not None and value < min_value:
        errors.append(f"{dotted_key} must be >= {min_value}")

    if max_value is not None and value > max_value:
        errors.append(f"{dotted_key} must be <= {max_value}")


def _validate_string_list(
    config: Dict[str, Any],
    dotted_key: str,
    *,
    errors: List[str],
    allow_empty: bool,
) -> None:
    found, value = _resolve(config, dotted_key)
    if not found:
        return

    if not isinstance(value, list):
        errors.append(f"{dotted_key} must be a list")
        return
    if not allow_empty and not value:
        errors.append(f"{dotted_key} must be a non-empty list")
        return
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{dotted_key} must contain only non-empty strings")


def _resolve(config: Dict[str, Any], dotted_key: str) -> Tuple[bool, Any]:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current
