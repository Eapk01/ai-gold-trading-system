"""Shared application service layer for CLI and GUI workflows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.ai_models import AIModelManager
from src.backtester import BacktestResult, Backtester
from src.broker_interface import BrokerManager
from src.config_utils import (
    ConfigValidationError,
    ensure_runtime_directories,
    get_target_column,
    load_config as load_validated_config,
    save_config,
)
from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.live_demo_trader import LiveDemoTrader
from src.model_tester import ModelTestResult, ModelTester
from src.report_store import ReportStore
from src.secret_store import BrokerSecretStore
from src.services import ReportWorkflowService, ResearchWorkflowService, TradingWorkflowService


CONFIG_PATH = "config/config.yaml"


@dataclass
class ServiceResponse:
    success: bool
    message: str
    data: Any = None
    artifacts: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_app_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate the application configuration."""
    return load_validated_config(config_path)


class ResearchAppService:
    """Shared service layer for the research workflow."""

    API_COMPATIBILITY_VERSION = "2026.04.config-sanity-v1"

    def __init__(self, config_path: str = CONFIG_PATH):
        logger.info("=== AI Gold Research System Startup ===")
        self.api_compatibility_version = self.API_COMPATIBILITY_VERSION
        self.config_path = config_path
        self.config = load_app_config(config_path)
        ensure_runtime_directories(self.config)
        self.secret_store = BrokerSecretStore()
        self.broker_manager = BrokerManager()

        self.feature_data = None
        self.selected_features: List[str] = []
        self.last_import_summary: Dict[str, Any] = {}
        self.loaded_model_path: Optional[str] = None
        self.latest_training_results: Dict[str, Any] = {}
        self.latest_backtest_summary: Dict[str, Any] = {}
        self.latest_backtest_artifacts: Dict[str, str] = {}
        self.latest_model_test_summary: Dict[str, Any] = {}
        self.latest_model_test_artifacts: Dict[str, str] = {}
        self.latest_data_preview: List[Dict[str, Any]] = []
        self.latest_model_analysis: Dict[str, Any] = {}
        self.report_store = ReportStore()

        self._migrate_plaintext_broker_secrets()
        self._build_runtime_components()
        self.research_workflows = ResearchWorkflowService(self)
        self.report_workflows = ReportWorkflowService(self, self.report_store)
        self.trading_workflows = TradingWorkflowService(self)

        self._load_saved_broker_profiles()
        self._autoconnect_saved_broker()
        self._autoload_latest_model()
        logger.info("Research system initialization complete")

    def _response(
        self,
        success: bool,
        message: str,
        *,
        data: Any = None,
        artifacts: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return ServiceResponse(
            success=success,
            message=message,
            data=data,
            artifacts=artifacts,
            errors=errors,
        ).to_dict()

    def _ensure_workflow_services(self) -> None:
        """Lazily attach delegated workflow helpers for lightweight test fixtures."""
        if not hasattr(self, "report_store"):
            self.report_store = ReportStore()
        if not hasattr(self, "research_workflows"):
            self.research_workflows = ResearchWorkflowService(self)
        if not hasattr(self, "report_workflows"):
            self.report_workflows = ReportWorkflowService(self, self.report_store)
        if not hasattr(self, "trading_workflows"):
            self.trading_workflows = TradingWorkflowService(self)

    def _load_saved_broker_profiles(self) -> None:
        profiles = self.config.get("brokers", {}).get("profiles", {})
        loaded_count = self.broker_manager.load_profiles(profiles)
        if loaded_count:
            logger.info(f"Loaded {loaded_count} saved broker profile(s)")

    def _migrate_plaintext_broker_secrets(self) -> None:
        changed = False

        exness_config = self.config.get("brokers", {}).get("exness", {})
        if "password" in exness_config and exness_config.get("password"):
            exness_config["password"] = ""
            changed = True

        profiles = self.config.get("brokers", {}).get("profiles", {})
        for profile_name, profile_data in profiles.items():
            password = str(profile_data.get("password", "")).strip()
            if password:
                self.secret_store.set_password(profile_name, password)
                profile_data["password"] = ""
                changed = True
                logger.info(f"Migrated plaintext broker secret for profile: {profile_name}")

        if changed:
            self._persist_config()

    def _autoconnect_saved_broker(self) -> None:
        profiles = self.config.get("brokers", {}).get("profiles", {})
        if not profiles:
            logger.info("No saved broker profiles available for auto-connect")
            return

        default_profile = str(self.config.get("brokers", {}).get("default_profile", "")).strip()
        candidates: List[str] = []
        if default_profile and default_profile in profiles:
            candidates.append(default_profile)

        for profile_name in profiles:
            if profile_name not in candidates:
                candidates.append(profile_name)

        for profile_name in candidates:
            try:
                secret = self.secret_store.get_password(profile_name)
                if not secret:
                    logger.warning(f"Auto-connect skipped for '{profile_name}': saved secret is missing")
                    continue
                if self.broker_manager.connect_broker(profile_name, password=secret):
                    self.config.setdefault("brokers", {})
                    self.config["brokers"]["default_profile"] = profile_name
                    self._persist_config()
                    logger.info(f"Auto-connected broker profile: {profile_name}")
                    return
            except Exception as exc:
                logger.warning(f"Auto-connect failed for broker profile '{profile_name}': {exc}")

        logger.warning("Saved broker profiles were loaded, but no broker auto-connect succeeded")

    def _persist_config(self) -> None:
        save_config(self.config, self.config_path)

    def _build_runtime_components(self) -> None:
        """Build all config-dependent runtime components from one config snapshot."""
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ai_model_manager = AIModelManager(self.config)
        self.backtester = Backtester(self.config)
        self.model_tester = ModelTester()
        self.auto_trader = LiveDemoTrader(
            self.config,
            self.broker_manager,
            self.feature_engineer,
            self.ai_model_manager,
        )

    def _reload_runtime_from_disk(self) -> None:
        """Reload config and rebuild all config-backed components together when safe."""
        if not getattr(self, "config_path", None):
            return
        if hasattr(self, "auto_trader") and self.auto_trader.get_status().get("running"):
            logger.info("Runtime reload skipped because the auto trader is currently running")
            return

        previous_model_path = self.loaded_model_path
        previous_selected_features = list(self.selected_features)

        self.config = load_app_config(self.config_path)
        ensure_runtime_directories(self.config)
        self._build_runtime_components()
        self._load_saved_broker_profiles()

        if previous_model_path and Path(previous_model_path).exists():
            if self.ai_model_manager.load_models(str(previous_model_path)):
                self.loaded_model_path = str(previous_model_path)
                self.selected_features = list(self.ai_model_manager.feature_columns) or previous_selected_features
            else:
                self.loaded_model_path = None
                self.selected_features = previous_selected_features
        else:
            self.selected_features = previous_selected_features

    def get_target_column(self) -> str:
        """Return the canonical target column used for train/eval workflows."""
        if hasattr(self, "ai_model_manager") and getattr(self.ai_model_manager, "target_column", ""):
            return str(self.ai_model_manager.target_column)
        return get_target_column(self.config)

    def _sanitize_model_name(self, model_name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.strip())
        return cleaned.strip("_") or "default"

    def _get_saved_model_files(self) -> List[Path]:
        return sorted(Path("models").glob("*.joblib"), key=lambda path: path.stat().st_mtime, reverse=True)

    def _serialize_backtest_result(self, result: BacktestResult) -> Dict[str, Any]:
        return {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "max_consecutive_wins": result.max_consecutive_wins,
            "max_consecutive_losses": result.max_consecutive_losses,
            "avg_winning_trade": result.avg_winning_trade,
            "avg_losing_trade": result.avg_losing_trade,
            "profit_factor": result.profit_factor,
            "start_date": result.start_date.isoformat() if result.start_date else None,
            "end_date": result.end_date.isoformat() if result.end_date else None,
            "equity_curve_points": len(result.equity_curve or []),
        }

    def _serialize_model_test_result(self, result: ModelTestResult) -> Dict[str, Any]:
        return dict(result.summary)

    def _autoload_latest_model(self) -> None:
        saved_models = self._get_saved_model_files()
        if not saved_models:
            return

        latest_model = saved_models[0]
        if self.ai_model_manager.load_models(str(latest_model)):
            self.loaded_model_path = str(latest_model)
            self.selected_features = list(self.ai_model_manager.feature_columns)
            logger.info(f"Auto-loaded latest saved model: {latest_model.name}")

    def import_and_prepare_data(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.research_workflows.import_and_prepare_data()

    def list_saved_models(self) -> Dict[str, Any]:
        models = []
        for model_path in self._get_saved_model_files():
            models.append(
                {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "modified_at": datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_loaded": self.loaded_model_path == str(model_path),
                }
            )

        message = "Saved models loaded" if models else "No saved model files found"
        return self._response(True, message, data=models)

    def load_saved_model(self, model_path_or_name: str) -> Dict[str, Any]:
        if not model_path_or_name:
            return self._response(False, "Model path or name is required")

        selected_model: Optional[Path] = None
        candidate_path = Path(model_path_or_name)
        if candidate_path.exists():
            selected_model = candidate_path
        else:
            for model_path in self._get_saved_model_files():
                if model_path.stem == model_path_or_name or model_path.name == model_path_or_name:
                    selected_model = model_path
                    break

        if selected_model is None:
            return self._response(False, f"Model not found: {model_path_or_name}")

        if self.ai_model_manager.load_models(str(selected_model)):
            self.loaded_model_path = str(selected_model)
            self.selected_features = list(self.ai_model_manager.feature_columns)
            return self._response(
                True,
                f"Loaded model: {selected_model.name}",
                data={
                    "name": selected_model.stem,
                    "path": str(selected_model),
                    "selected_features": list(self.selected_features),
                },
            )

        return self._response(False, f"Failed to load model: {selected_model.name}")

    def train_models(self, model_name: str = "default") -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.research_workflows.train_models(model_name)

    def run_backtest(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.research_workflows.run_backtest()

    def run_model_test(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.research_workflows.run_model_test()

    def list_backtest_reports(self, limit: int = 10) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.report_workflows.list_backtest_reports(limit)

    def get_backtest_report(self, report_path: str) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.report_workflows.get_backtest_report(report_path)

    def list_model_test_reports(self, limit: int = 10) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.report_workflows.list_model_test_reports(limit)

    def get_model_test_report(self, report_path: str) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.report_workflows.get_model_test_report(report_path)

    def get_model_analysis(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.research_workflows.get_model_analysis()

    def get_system_status(self) -> Dict[str, Any]:
        broker_status = self.broker_manager.get_broker_status()
        status = {
            "last_import_summary": self.last_import_summary,
            "models_trained": len(self.ai_model_manager.models),
            "loaded_model_file": Path(self.loaded_model_path).name if self.loaded_model_path else None,
            "loaded_model_path": self.loaded_model_path,
            "saved_model_files": len(self._get_saved_model_files()),
            "saved_broker_profiles": len(self.config.get("brokers", {}).get("profiles", {})),
            "active_broker": broker_status.get("active_broker"),
            "selected_features": len(self.selected_features),
            "latest_backtest_summary": self.latest_backtest_summary,
            "latest_backtest_artifacts": self.latest_backtest_artifacts,
            "latest_model_test_summary": self.latest_model_test_summary,
            "latest_model_test_artifacts": self.latest_model_test_artifacts,
            "auto_trader": self.auto_trader.get_status(),
        }
        return self._response(True, "System status loaded", data=status)

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        system_status = self.get_system_status().get("data") or {}
        config_summary = self.get_configuration_summary().get("data") or {}
        trading_snapshot = self.get_trading_snapshot().get("data") or {}

        account = trading_snapshot.get("account") or {}
        positions = trading_snapshot.get("positions") or []
        auto_trader = system_status.get("auto_trader") or {}

        open_positions_profit_total = 0.0
        open_buy_positions = 0
        open_sell_positions = 0
        normalized_positions = []

        for position in positions:
            profit = self._safe_float(position.get("profit")) or 0.0
            position_type = self._safe_int(position.get("type"))
            side = "Buy" if position_type == 0 else "Sell"
            if side == "Buy":
                open_buy_positions += 1
            else:
                open_sell_positions += 1
            open_positions_profit_total += profit
            normalized_positions.append(
                {
                    "Ticket": str(position.get("ticket", "")),
                    "Symbol": str(position.get("symbol", "")),
                    "Side": side,
                    "Volume": self._safe_float(position.get("volume")),
                    "Current Profit": profit,
                }
            )

        payload = {
            "broker": {
                "active_broker": system_status.get("active_broker"),
                "broker_connected": bool(trading_snapshot.get("broker_connected")),
                "balance": self._safe_float(account.get("balance")),
                "equity": self._safe_float(account.get("equity")),
                "margin_free": self._safe_float(account.get("margin_free")),
                "leverage": self._safe_float(account.get("leverage")),
            },
            "live_trading": {
                "symbol": config_summary.get("trading_symbol"),
                "timeframe": config_summary.get("timeframe"),
                "running": bool(auto_trader.get("running")),
                "market_state": auto_trader.get("market_state"),
                "latest_action": auto_trader.get("latest_action"),
                "last_processed_candle": auto_trader.get("last_processed_candle"),
                "managed_positions": self._safe_int(auto_trader.get("managed_positions")),
                "last_candle_age_seconds": auto_trader.get("last_candle_age_seconds"),
            },
            "positions": {
                "open_positions_count": len(positions),
                "open_positions_profit_total": open_positions_profit_total,
                "open_buy_positions": open_buy_positions,
                "open_sell_positions": open_sell_positions,
                "items": normalized_positions,
            },
            "research": {
                "loaded_model_file": system_status.get("loaded_model_file"),
                "saved_model_files": system_status.get("saved_model_files", 0),
                "selected_features": system_status.get("selected_features", 0),
                "dataset_imported": bool(system_status.get("last_import_summary")),
                "last_import_summary": system_status.get("last_import_summary") or {},
                "latest_backtest_summary": system_status.get("latest_backtest_summary") or {},
                "latest_backtest_artifacts": system_status.get("latest_backtest_artifacts") or {},
                "latest_model_test_summary": system_status.get("latest_model_test_summary") or {},
                "latest_model_test_artifacts": system_status.get("latest_model_test_artifacts") or {},
            },
        }
        return self._response(True, "Dashboard snapshot loaded", data=payload)

    def get_configuration_summary(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.get_configuration_summary()

    def get_trading_snapshot(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        bars: int = 200,
    ) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.get_trading_snapshot(symbol=symbol, timeframe=timeframe, bars=bars)

    def place_manual_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.place_manual_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def close_manual_position(self, position_ticket: str) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.close_manual_position(position_ticket)

    def list_broker_profiles(self) -> Dict[str, Any]:
        status = self.broker_manager.get_broker_status()
        active_broker = status.get("active_broker")
        profiles = []
        for name, info in status.items():
            if name == "active_broker":
                continue
            profiles.append(
                {
                    "name": name,
                    "type": info.get("type"),
                    "connected": info.get("connected"),
                    "sandbox": info.get("sandbox"),
                    "secret_saved": self.secret_store.has_password(name),
                    "last_heartbeat": info.get("last_heartbeat"),
                    "is_active": name == active_broker,
                }
            )
        return self._response(True, "Broker profiles loaded", data=profiles)

    def save_broker_profile(
        self,
        *,
        name: str,
        login: str,
        password: str,
        server: str,
        terminal_path: str = "",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.save_broker_profile(
            name=name,
            login=login,
            password=password,
            server=server,
            terminal_path=terminal_path,
            overwrite=overwrite,
        )

    def connect_broker(self, name: str) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.connect_broker(name)

    def disconnect_all_brokers(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.disconnect_all_brokers()

    def delete_broker_profile(self, name: str) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.delete_broker_profile(name)

    def start_auto_trader(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.start_auto_trader()

    def stop_auto_trader(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.stop_auto_trader()

    def get_auto_trader_status(self) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.get_auto_trader_status()

    def get_auto_trader_events(self, limit: int = 20) -> Dict[str, Any]:
        self._ensure_workflow_services()
        return self.trading_workflows.get_auto_trader_events(limit)

    def _get_chart_data(self, symbol: str, timeframe: str, bars: int) -> tuple[List[Dict[str, Any]], str, str]:
        broker_chart = self.broker_manager.get_historical_data(symbol, timeframe, bars)
        if broker_chart:
            return broker_chart, "broker", "Loaded directly from the active MT5 broker connection."

        broker_reason = "MT5 chart data was unavailable."
        if not self.broker_manager.active_broker:
            broker_reason = "No active broker connection, so the chart fell back to the local dataset."

        try:
            local_data = self.data_collector.load_data_from_db("raw_data")
            if local_data.empty:
                _, local_data = self.data_collector.import_default_dataset()
        except Exception as exc:
            logger.warning(f"Failed to load local chart data: {exc}")
            local_data = pd.DataFrame()

        if local_data.empty:
            return [], "unavailable", f"{broker_reason} No local dataset was available either."

        frame = local_data.reset_index().copy()
        if "Timestamp" not in frame.columns and "DateTime" in frame.columns:
            frame["Timestamp"] = pd.to_datetime(frame["DateTime"], errors="coerce")
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
        frame = frame.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).tail(int(bars))

        chart = [
            {
                "timestamp": int(row.Timestamp.timestamp()),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "volume": float(getattr(row, "Volume", 0.0)),
            }
            for row in frame.itertuples(index=False)
        ]
        return chart, "local_dataset", broker_reason

    def cleanup(self) -> None:
        try:
            self.auto_trader.stop()
            self.broker_manager.disconnect_all()
        except Exception as exc:
            logger.warning(f"Cleanup warning while disconnecting brokers: {exc}")

    def _safe_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
