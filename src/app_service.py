"""
Shared application service layer for CLI and GUI workflows.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.ai_models import AIModelManager
from src.backtester import BacktestResult, Backtester
from src.broker_interface import BrokerManager, broker_config_to_dict, create_broker_config
from src.config_utils import (
    ConfigValidationError,
    ensure_runtime_directories,
    load_config as load_validated_config,
    save_config,
)
from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.live_demo_trader import LiveDemoTrader
from src.secret_store import BrokerSecretStore


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

    def __init__(self, config_path: str = CONFIG_PATH):
        logger.info("=== AI Gold Research System Startup ===")
        self.config_path = config_path
        self.config = load_app_config(config_path)
        ensure_runtime_directories(self.config)
        self.secret_store = BrokerSecretStore()
        self._migrate_plaintext_broker_secrets()

        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ai_model_manager = AIModelManager(self.config)
        self.backtester = Backtester(self.config)
        self.broker_manager = BrokerManager()
        self.auto_trader = LiveDemoTrader(
            self.config,
            self.broker_manager,
            self.feature_engineer,
            self.ai_model_manager,
        )

        self.feature_data = None
        self.selected_features: List[str] = []
        self.last_import_summary: Dict[str, Any] = {}
        self.loaded_model_path: Optional[str] = None
        self.latest_training_results: Dict[str, Any] = {}
        self.latest_backtest_summary: Dict[str, Any] = {}
        self.latest_backtest_artifacts: Dict[str, str] = {}
        self.latest_data_preview: List[Dict[str, Any]] = []
        self.latest_model_analysis: Dict[str, Any] = {}

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
        logger.info("Starting local dataset import and preparation...")
        try:
            csv_path, historical_data = self.data_collector.import_default_dataset()
            is_valid, issues = self.data_collector.validate_data_quality(historical_data)

            if historical_data.empty:
                logger.error("Imported dataset is empty after normalization")
                return self._response(False, "Imported dataset is empty after normalization")

            self.data_collector.save_data_to_db(historical_data, "raw_data")
            self.feature_data = self.feature_engineer.create_feature_matrix(historical_data, include_targets=True)

            if self.feature_data.empty:
                logger.error("Failed to create features")
                return self._response(False, "Failed to create features")

            self.selected_features = self.feature_engineer.select_features(
                self.feature_data,
                target_column="Future_Direction_1",
                method="correlation",
                max_features=30,
            )

            if not self.selected_features:
                logger.error("Feature selection failed")
                return self._response(False, "Feature selection failed")

            self.last_import_summary = {
                "path": str(csv_path),
                "rows": len(historical_data),
                "selected_features": len(self.selected_features),
                "data_valid": is_valid,
                "issues": issues,
                "feature_rows": len(self.feature_data),
            }
            preview_frame = historical_data.reset_index().head(20)
            self.latest_data_preview = preview_frame.to_dict(orient="records")

            logger.info(
                f"Local dataset preparation complete - {len(historical_data)} raw rows, "
                f"{len(self.selected_features)} selected features"
            )
            return self._response(
                True,
                f"Imported {len(historical_data)} rows and selected {len(self.selected_features)} features",
                data={
                    "summary": self.last_import_summary,
                    "selected_features": list(self.selected_features),
                    "preview": self.latest_data_preview,
                },
                errors=list(issues),
            )
        except Exception as exc:
            logger.error(f"Failed to import local dataset: {exc}")
            return self._response(False, f"Local dataset import failed: {exc}", errors=[str(exc)])

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
        if self.feature_data is None or not self.selected_features:
            logger.error("Please prepare the data first")
            return self._response(False, "Please prepare the data first")

        logger.info("Starting AI model training...")
        training_results = self.ai_model_manager.train_ensemble_models(
            self.feature_data,
            self.selected_features,
            target_column="Future_Direction_1",
        )

        if not training_results:
            logger.error("Model training failed")
            return self._response(False, "Model training failed")

        safe_name = self._sanitize_model_name(model_name or "default")
        model_path = Path("models") / f"{safe_name}.joblib"
        self.ai_model_manager.save_models(model_path)
        self.loaded_model_path = str(model_path)
        self.selected_features = list(self.ai_model_manager.feature_columns)
        self.latest_training_results = training_results

        return self._response(
            True,
            f"Training complete. Saved model file: {model_path.name}",
            data={
                "training_results": training_results,
                "saved_model_name": model_path.stem,
                "saved_model_path": str(model_path),
                "selected_features": list(self.selected_features),
            },
            artifacts={"model_path": str(model_path)},
        )

    def run_backtest(self) -> Dict[str, Any]:
        logger.info("Starting professional backtest...")

        if not self.ai_model_manager.models:
            logger.error("Please train or load the models first")
            return self._response(False, "Please train or load the models first")
        if self.feature_data is None or not self.selected_features:
            logger.error("Please prepare the data first")
            return self._response(False, "Please prepare the data first")

        try:
            result = self.backtester.run_backtest(
                self.feature_data,
                self.ai_model_manager,
                self.selected_features,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"reports/backtest_result_{timestamp}.json"
            chart_file = f"reports/backtest_chart_{timestamp}.png"
            summary_file = ""

            self.backtester.save_results(result, result_file)
            self.backtester.plot_results(result, chart_file)

            trade_summary = self.backtester.get_trade_summary()
            if not trade_summary.empty:
                summary_file = f"reports/trade_summary_{timestamp}.csv"
                trade_summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
                logger.info(f"Trade summary saved: {summary_file}")

            result_summary = self._serialize_backtest_result(result)
            artifacts = {
                "report_file": result_file,
                "chart_file": chart_file,
            }
            if summary_file:
                artifacts["trade_summary_file"] = summary_file

            self.latest_backtest_summary = result_summary
            self.latest_backtest_artifacts = artifacts

            return self._response(
                True,
                "Backtest completed successfully",
                data={
                    "summary": result_summary,
                    "trade_summary_rows": len(trade_summary),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Professional backtest failed: {exc}")
            return self._response(False, f"Professional backtest failed: {exc}", errors=[str(exc)])

    def list_backtest_reports(self, limit: int = 10) -> Dict[str, Any]:
        report_files = sorted(glob.glob("reports/backtest_result_*.json"), reverse=True)
        reports = [
            {
                "path": file_path,
                "name": Path(file_path).name,
                "timestamp": Path(file_path).stem.replace("backtest_result_", ""),
            }
            for file_path in report_files[:limit]
        ]
        message = "Backtest reports loaded" if reports else "No backtest report files found"
        return self._response(True, message, data=reports)

    def get_backtest_report(self, report_path: str) -> Dict[str, Any]:
        try:
            with open(report_path, "r", encoding="utf-8") as report_file:
                report = json.load(report_file)

            summary = report.get("backtest_summary", {})
            return self._response(
                True,
                f"Loaded backtest report: {Path(report_path).name}",
                data={
                    "path": report_path,
                    "summary": summary,
                    "trade_count": len(report.get("trades", [])),
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load backtest report: {exc}")
            return self._response(False, f"Failed to load backtest report: {exc}", errors=[str(exc)])

    def get_model_analysis(self) -> Dict[str, Any]:
        if not self.ai_model_manager.models:
            return self._response(False, "Please train or load the models first")

        summary = self.ai_model_manager.get_models_summary()
        feature_importance = {}
        for model_name in self.ai_model_manager.models:
            importance = self.ai_model_manager.get_feature_importance(model_name)
            if importance:
                feature_importance[model_name] = list(importance.items())[:10]

        self.latest_model_analysis = {
            "summary": summary.to_dict(orient="records") if not summary.empty else [],
            "feature_importance": feature_importance,
        }
        return self._response(True, "Model analysis loaded", data=self.latest_model_analysis)

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
            },
        }
        return self._response(True, "Dashboard snapshot loaded", data=payload)

    def get_configuration_summary(self) -> Dict[str, Any]:
        summary = {
            "trading_symbol": self.config["trading"]["symbol"],
            "timeframe": self.config["trading"]["timeframe"],
            "primary_data_source": self.config["data_sources"]["primary"],
            "dataset_directory": self.config["data_sources"].get("dataset_directory", "data/imports"),
            "model_type": self.config["ai_model"]["type"],
            "enabled_models": list(self.config["ai_model"]["models"]),
            "database_path": self.config["database"]["path"],
        }
        return self._response(True, "Configuration loaded", data=summary)

    def get_trading_snapshot(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        bars: int = 200,
    ) -> Dict[str, Any]:
        trading_symbol = (symbol or self.config["trading"]["symbol"]).strip()
        trading_timeframe = (timeframe or self.config["trading"]["timeframe"]).strip()

        quote = self.broker_manager.get_market_data(trading_symbol)
        account_info = self.broker_manager.get_account_info()
        positions = self.broker_manager.get_positions()
        chart_data, chart_source, chart_reason = self._get_chart_data(trading_symbol, trading_timeframe, bars)

        return self._response(
            True,
            f"Trading snapshot loaded for {trading_symbol}",
            data={
                "symbol": trading_symbol,
                "timeframe": trading_timeframe,
                "quote": quote,
                "account": account_info,
                "positions": positions,
                "chart": chart_data,
                "chart_source": chart_source,
                "chart_reason": chart_reason,
                "broker_connected": bool(self.broker_manager.active_broker),
            },
        )

    def place_manual_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Dict[str, Any]:
        cleaned_symbol = symbol.strip() or self.config["trading"]["symbol"]
        if quantity <= 0:
            return self._response(False, "Quantity must be greater than zero")

        result = self.broker_manager.place_order(
            symbol=cleaned_symbol,
            side=side.strip().lower(),
            quantity=float(quantity),
            order_type="market",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if result.get("success"):
            return self._response(
                True,
                f"{side.title()} market order submitted for {cleaned_symbol}",
                data=result,
            )

        return self._response(
            False,
            result.get("error", f"Failed to submit {side} market order"),
            data=result,
            errors=[result.get("error")] if result.get("error") else None,
        )

    def close_manual_position(self, position_ticket: str) -> Dict[str, Any]:
        if not str(position_ticket).strip():
            return self._response(False, "Position ticket is required")

        result = self.broker_manager.close_position(str(position_ticket).strip())
        if result.get("success"):
            return self._response(
                True,
                f"Closed position {position_ticket}",
                data=result,
            )

        return self._response(
            False,
            result.get("error", f"Failed to close position {position_ticket}"),
            data=result,
            errors=[result.get("error")] if result.get("error") else None,
        )

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
        name = name.strip()
        if not name:
            return self._response(False, "Profile name is required")
        if not str(password).strip():
            return self._response(False, "MT5 password is required")

        existing_profiles = self.config.get("brokers", {}).get("profiles", {})
        if name in existing_profiles and not overwrite:
            return self._response(False, f"A saved profile named '{name}' already exists")

        try:
            broker_config = create_broker_config(
                broker_type="exness",
                login=login.strip(),
                password="",
                server=server.strip(),
                terminal_path=terminal_path.strip(),
                sandbox=False,
            )
        except Exception as exc:
            return self._response(False, f"Failed to create broker profile: {exc}", errors=[str(exc)])

        success = self.broker_manager.add_broker(name, broker_config)
        if not success:
            return self._response(False, f"Failed to save broker profile: {name}")

        self.config.setdefault("brokers", {})
        self.config["brokers"].setdefault("profiles", {})
        self.config["brokers"]["profiles"][name] = broker_config_to_dict(broker_config)
        if not self.config["brokers"].get("default_profile"):
            self.config["brokers"]["default_profile"] = name
        self.secret_store.set_password(name, password)
        self._persist_config()

        return self._response(
            True,
            f"Broker profile saved securely: {name}",
            data={"name": name, "server": broker_config.server},
        )

    def connect_broker(self, name: str) -> Dict[str, Any]:
        secret = self.secret_store.get_password(name)
        if not secret:
            return self._response(False, f"Connection failed for broker: {name}. Saved secret is missing.")
        success = self.broker_manager.connect_broker(name, password=secret)
        if not success:
            return self._response(False, f"Connection failed for broker: {name}")
        self.config.setdefault("brokers", {})
        self.config["brokers"]["default_profile"] = name
        self._persist_config()
        return self._response(True, f"Connection successful for broker: {name}")

    def disconnect_all_brokers(self) -> Dict[str, Any]:
        self.broker_manager.disconnect_all()
        return self._response(True, "Disconnected all broker connections")

    def delete_broker_profile(self, name: str) -> Dict[str, Any]:
        profiles = self.config.setdefault("brokers", {}).setdefault("profiles", {})
        if name not in profiles:
            return self._response(False, f"Saved profile not found: {name}")

        del profiles[name]
        self.broker_manager.remove_broker(name)
        self.secret_store.delete_password(name)
        if self.config["brokers"].get("default_profile") == name:
            self.config["brokers"]["default_profile"] = next(iter(profiles.keys()), "")
        self._persist_config()
        return self._response(True, f"Broker profile deleted: {name}")

    def start_auto_trader(self) -> Dict[str, Any]:
        success, message = self.auto_trader.start()
        return self._response(success, message, data=self.auto_trader.get_status())

    def stop_auto_trader(self) -> Dict[str, Any]:
        success, message = self.auto_trader.stop()
        return self._response(success, message, data=self.auto_trader.get_status())

    def get_auto_trader_status(self) -> Dict[str, Any]:
        return self._response(True, "Auto trader status loaded", data=self.auto_trader.get_status())

    def get_auto_trader_events(self, limit: int = 20) -> Dict[str, Any]:
        return self._response(
            True,
            "Auto trader events loaded",
            data=self.auto_trader.get_recent_events(limit),
        )

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
