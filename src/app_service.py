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

        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ai_model_manager = AIModelManager(self.config)
        self.backtester = Backtester(self.config)
        self.broker_manager = BrokerManager()

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
            "latest_backtest_artifacts": self.latest_backtest_artifacts,
        }
        return self._response(True, "System status loaded", data=status)

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

        existing_profiles = self.config.get("brokers", {}).get("profiles", {})
        if name in existing_profiles and not overwrite:
            return self._response(False, f"A saved profile named '{name}' already exists")

        try:
            broker_config = create_broker_config(
                broker_type="exness",
                login=login.strip(),
                password=password,
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
        self._persist_config()

        return self._response(
            True,
            f"Broker profile saved: {name}",
            data={"name": name, "server": broker_config.server},
        )

    def connect_broker(self, name: str) -> Dict[str, Any]:
        success = self.broker_manager.connect_broker(name)
        if not success:
            return self._response(False, f"Connection failed for broker: {name}")
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
        if self.config["brokers"].get("default_profile") == name:
            self.config["brokers"]["default_profile"] = next(iter(profiles.keys()), "")
        self._persist_config()
        return self._response(True, f"Broker profile deleted: {name}")

    def cleanup(self) -> None:
        try:
            self.broker_manager.disconnect_all()
        except Exception as exc:
            logger.warning(f"Cleanup warning while disconnecting brokers: {exc}")
