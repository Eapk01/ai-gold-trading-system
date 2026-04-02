"""
Focused v1 entry point for dataset import, model training, backtesting, and Exness setup.
"""

import glob
import json
import os
import sys
from datetime import datetime

from loguru import logger

sys.path.append("src")

from src.ai_models import AIModelManager
from src.backtester import Backtester
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


def load_config():
    """Load and validate the application configuration."""
    try:
        return load_validated_config(CONFIG_PATH)
    except ConfigValidationError as exc:
        logger.error(f"Configuration validation failed: {exc}")
        return None
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        return None


def setup_directories(config=None):
    """Create runtime directories used by the application."""
    if config:
        ensure_runtime_directories(config)
        return

    for directory in ["data", "logs", "models", "reports"]:
        os.makedirs(directory, exist_ok=True)


def setup_logging():
    """Configure application logging."""
    logger.add(
        "logs/trading_system_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )


class TradingSystem:
    """Focused v1 orchestration shell for research and broker setup."""

    def __init__(self):
        logger.info("=== AI Gold Research System Startup ===")

        self.config = load_config()
        if not self.config:
            raise RuntimeError("Failed to load configuration")

        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ai_model_manager = AIModelManager(self.config)
        self.backtester = Backtester(self.config)
        self.broker_manager = BrokerManager()

        self.feature_data = None
        self.selected_features = []
        self.last_import_summary = {}

        self._load_saved_broker_profiles()
        logger.info("Research system initialization complete")

    def _load_saved_broker_profiles(self):
        """Load saved broker profiles from config into memory."""
        profiles = self.config.get("brokers", {}).get("profiles", {})
        loaded_count = self.broker_manager.load_profiles(profiles)
        if loaded_count:
            logger.info(f"Loaded {loaded_count} saved broker profile(s)")

    def _persist_config(self):
        """Persist the active config file."""
        save_config(self.config, CONFIG_PATH)

    def _persist_broker_profile(self, name: str, broker_config):
        """Persist a broker profile in config."""
        self.config.setdefault("brokers", {})
        self.config["brokers"].setdefault("profiles", {})
        self.config["brokers"]["profiles"][name] = broker_config_to_dict(broker_config)
        if not self.config["brokers"].get("default_profile"):
            self.config["brokers"]["default_profile"] = name
        self._persist_config()

    def _delete_broker_profile(self, name: str) -> bool:
        """Delete a broker profile from config and manager state."""
        profiles = self.config.setdefault("brokers", {}).setdefault("profiles", {})
        if name not in profiles:
            return False

        del profiles[name]
        self.broker_manager.remove_broker(name)
        if self.config["brokers"].get("default_profile") == name:
            self.config["brokers"]["default_profile"] = next(iter(profiles.keys()), "")
        self._persist_config()
        return True

    def collect_and_prepare_data(self):
        """Import a local dataset and prepare the feature matrix."""
        logger.info("Starting local dataset import and preparation...")

        dataset_directory = self.config.get("data_sources", {}).get("dataset_directory", "data/imports")
        print("\n📁 Local Dataset Import")
        print("=" * 50)
        print(f"Default dataset directory: {dataset_directory}")

        try:
            csv_path, historical_data = self.data_collector.import_default_dataset()
            print(f"Using dataset: {csv_path}")
        except Exception as exc:
            logger.error(f"Failed to import local dataset: {exc}")
            print(f"❌ Local dataset import failed: {exc}")
            return False

        is_valid, issues = self.data_collector.validate_data_quality(historical_data)
        if issues:
            logger.warning(f"Data quality issues: {issues}")
            print("\n⚠️ Data quality warnings:")
            for issue in issues:
                print(f" - {issue}")

        if historical_data.empty:
            logger.error("Imported dataset is empty after normalization")
            return False

        self.data_collector.save_data_to_db(historical_data, "raw_data")
        self.feature_data = self.feature_engineer.create_feature_matrix(historical_data, include_targets=True)

        if self.feature_data.empty:
            logger.error("Failed to create features")
            return False

        self.selected_features = self.feature_engineer.select_features(
            self.feature_data,
            target_column="Future_Direction_1",
            method="correlation",
            max_features=30,
        )

        if not self.selected_features:
            logger.error("Feature selection failed")
            return False

        self.last_import_summary = {
            "path": csv_path,
            "rows": len(historical_data),
            "selected_features": len(self.selected_features),
            "data_valid": is_valid,
        }
        logger.info(
            f"Local dataset preparation complete - {len(historical_data)} raw rows, "
            f"{len(self.selected_features)} selected features"
        )
        print(f"✅ Imported {len(historical_data)} rows and selected {len(self.selected_features)} features")
        return True

    def train_models(self):
        """Train the configured ML models."""
        if self.feature_data is None or not self.selected_features:
            logger.error("Please prepare the data first")
            return False

        logger.info("Starting AI model training...")
        training_results = self.ai_model_manager.train_ensemble_models(
            self.feature_data,
            self.selected_features,
            target_column="Future_Direction_1",
        )

        if not training_results:
            logger.error("Model training failed")
            return False

        model_path = f"models/ai_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        self.ai_model_manager.save_models(model_path)

        print("\n=== Model Training Results ===")
        for model_name, performance in training_results.items():
            accuracy = performance.get("test_accuracy", 0)
            print(f"{model_name}: test accuracy = {accuracy:.4f}")

        return True

    def run_backtest(self):
        """Run a backtest on the currently prepared feature dataset."""
        logger.info("Starting professional backtest...")

        if not self.ai_model_manager.models:
            logger.error("Please train the models first")
            return False
        if self.feature_data is None or not self.selected_features:
            logger.error("Please prepare the data first")
            return False

        try:
            result = self.backtester.run_backtest(
                self.feature_data,
                self.ai_model_manager,
                self.feature_engineer,
                self.selected_features,
            )

            print("\n=== Backtest Result Summary ===")
            print(f"Total trades: {result.total_trades}")
            print(f"Winning trades: {result.winning_trades}")
            print(f"Losing trades: {result.losing_trades}")
            print(f"Win rate: {result.win_rate:.1%}")
            print(f"Total PnL: ${result.total_pnl:.2f}")
            print(f"Max drawdown: {result.max_drawdown:.2%}")
            print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
            print(f"Profit factor: {result.profit_factor:.2f}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"reports/backtest_result_{timestamp}.json"
            self.backtester.save_results(result, result_file)

            chart_file = f"reports/backtest_chart_{timestamp}.png"
            self.backtester.plot_results(result, chart_file)

            trade_summary = self.backtester.get_trade_summary()
            if not trade_summary.empty:
                summary_file = f"reports/trade_summary_{timestamp}.csv"
                trade_summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
                logger.info(f"Trade summary saved: {summary_file}")

            return True
        except Exception as exc:
            logger.error(f"Professional backtest failed: {exc}")
            return False

    def show_backtest_reports(self):
        """View recent backtest result files."""
        print("\n=== Backtest Reports ===")
        report_files = sorted(glob.glob("reports/backtest_result_*.json"), reverse=True)
        if not report_files:
            print("No backtest report files found")
            return

        print("Available backtest reports:")
        for index, file_path in enumerate(report_files[:5], 1):
            timestamp = file_path.split("_")[-1].replace(".json", "")
            print(f"{index}. {timestamp}")

        choice = input("\nSelect a report number (1-5): ").strip()
        if not choice.isdigit():
            print("❌ Invalid selection")
            return

        report_index = int(choice) - 1
        if not 0 <= report_index < min(5, len(report_files)):
            print("❌ Invalid selection")
            return

        with open(report_files[report_index], "r", encoding="utf-8") as report_file:
            report = json.load(report_file)

        summary = report["backtest_summary"]
        print("\n=== Backtest Report Details ===")
        print(f"Total trades: {summary['total_trades']}")
        print(f"Winning trades: {summary['winning_trades']}")
        print(f"Losing trades: {summary['losing_trades']}")
        print(f"Win rate: {summary['win_rate']:.1%}")
        print(f"Total PnL: ${summary['total_pnl']:.2f}")
        print(f"Max drawdown: {summary['max_drawdown']:.2%}")
        print(f"Sharpe ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Profit factor: {summary['profit_factor']:.2f}")

    def analyze_model_performance(self):
        """Inspect trained model metrics and feature importance."""
        print("\n=== Model Performance Analysis ===")
        if not self.ai_model_manager.models:
            print("Please train the models first")
            return

        summary = self.ai_model_manager.get_models_summary()
        if not summary.empty:
            columns = [column for column in ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1"] if column in summary.columns]
            print("\nModel performance comparison:")
            print(summary[columns].to_string(index=False))

        print("\n=== Feature Importance Analysis ===")
        for model_name in self.ai_model_manager.models:
            importance = self.ai_model_manager.get_feature_importance(model_name)
            if importance:
                print(f"\n{model_name} top features:")
                for index, (feature, score) in enumerate(list(importance.items())[:10], 1):
                    print(f"{index:2d}. {feature}: {score:.4f}")

    def show_system_status(self):
        """Display a compact status overview of the focused v1 workflow."""
        print("\n=== System Status ===")
        if self.last_import_summary:
            print(f"Last dataset: {self.last_import_summary.get('path')}")
            print(f"Imported rows: {self.last_import_summary.get('rows', 0)}")
            print(f"Selected features: {self.last_import_summary.get('selected_features', 0)}")
        else:
            print("Dataset: not imported yet")

        print(f"Models trained: {len(self.ai_model_manager.models)}")
        print(f"Saved broker profiles: {len(self.config.get('brokers', {}).get('profiles', {}))}")

        broker_status = self.broker_manager.get_broker_status()
        print(f"Active broker: {broker_status.get('active_broker', 'None')}")

    def broker_interface_management(self):
        """Manage saved Exness broker profiles and connectivity."""
        try:
            print("\n🏦 Exness Broker Management")
            print("=" * 50)

            while True:
                print("\nBroker options:")
                print("1. Add saved Exness profile")
                print("2. View broker status")
                print("3. Connect saved broker")
                print("4. Disconnect all brokers")
                print("5. Delete saved broker")
                print("0. Return to main menu")

                choice = input("Select an action: ").strip()

                if choice == "0":
                    break
                if choice == "1":
                    name = input("Enter profile name: ").strip()
                    if not name:
                        print("❌ Profile name is required")
                        continue

                    existing_profiles = self.config.get("brokers", {}).get("profiles", {})
                    if name in existing_profiles:
                        overwrite = input("A saved profile with this name already exists. Overwrite it? (y/N): ").strip().lower()
                        if overwrite != "y":
                            print("Profile save canceled")
                            continue

                    broker_config = create_broker_config(
                        broker_type="exness",
                        login=input("MT5 Login: ").strip(),
                        password=input("MT5 Password: ").strip(),
                        server=input("MT5 Server: ").strip(),
                        terminal_path=input("MT5 Terminal Path (optional): ").strip(),
                        sandbox=False,
                    )

                    success = self.broker_manager.add_broker(name, broker_config)
                    if success:
                        self._persist_broker_profile(name, broker_config)
                    print(f"{'✅' if success else '❌'} Broker profile {'saved' if success else 'failed'}")
                    continue

                if choice == "2":
                    status = self.broker_manager.get_broker_status()
                    print("\n📋 Broker Status:")
                    for name, info in status.items():
                        if name == "active_broker":
                            continue
                        active_label = " (active)" if status.get("active_broker") == name else ""
                        connection_label = "connected" if info.get("connected") else "saved"
                        print(f"{name}{active_label}: type={info.get('type')} status={connection_label}")
                    print(f"Active broker: {status.get('active_broker', 'None')}")
                    continue

                if choice == "3":
                    status = self.broker_manager.get_broker_status()
                    brokers = [name for name in status.keys() if name != "active_broker"]
                    if not brokers:
                        print("❌ No brokers configured")
                        continue

                    print("Available brokers:")
                    for index, broker_name in enumerate(brokers, 1):
                        print(f"{index}. {broker_name}")

                    try:
                        broker_index = int(input("Select broker (number): ")) - 1
                    except ValueError:
                        print("❌ Please enter a valid number")
                        continue

                    if not 0 <= broker_index < len(brokers):
                        print("❌ Invalid selection")
                        continue

                    broker_name = brokers[broker_index]
                    success = self.broker_manager.connect_broker(broker_name)
                    print(f"{'✅' if success else '❌'} Connection {'successful' if success else 'failed'}")
                    continue

                if choice == "4":
                    self.broker_manager.disconnect_all()
                    print("✅ Disconnected all broker connections")
                    continue

                if choice == "5":
                    saved_profiles = list(self.config.get("brokers", {}).get("profiles", {}).keys())
                    if not saved_profiles:
                        print("❌ No saved broker profiles found")
                        continue

                    print("Saved broker profiles:")
                    for index, broker_name in enumerate(saved_profiles, 1):
                        print(f"{index}. {broker_name}")

                    try:
                        broker_index = int(input("Select broker to delete (number): ")) - 1
                    except ValueError:
                        print("❌ Please enter a valid number")
                        continue

                    if not 0 <= broker_index < len(saved_profiles):
                        print("❌ Invalid selection")
                        continue

                    broker_name = saved_profiles[broker_index]
                    confirm = input(f"Delete saved profile '{broker_name}'? (y/N): ").strip().lower()
                    if confirm == "y" and self._delete_broker_profile(broker_name):
                        print("✅ Broker profile deleted")
                    else:
                        print("Deletion canceled")
                    continue

                print("❌ Invalid selection")
        except Exception as exc:
            print(f"❌ Broker interface management failed: {exc}")
            logger.error(f"Broker interface management error: {exc}")

    def system_configuration(self):
        """Display the active runtime configuration for the focused workflow."""
        print("\n=== System Configuration ===")
        print(f"Trading symbol: {self.config['trading']['symbol']}")
        print(f"Timeframe: {self.config['trading']['timeframe']}")
        print(f"Primary data source: {self.config['data_sources']['primary']}")
        print(f"Dataset directory: {self.config['data_sources'].get('dataset_directory', 'data/imports')}")
        print(f"Model type: {self.config['ai_model']['type']}")
        print(f"Enabled models: {', '.join(self.config['ai_model']['models'])}")
        print(f"Database path: {self.config['database']['path']}")

    def welcome(self):
        """Display the startup banner."""
        print("\n" + "=" * 60)
        print("AI Gold Research System")
        print("Local dataset import, model training, backtesting, and Exness setup")
        print("=" * 60)

    def cleanup(self):
        """Disconnect broker sessions before exit."""
        try:
            self.broker_manager.disconnect_all()
        except Exception as exc:
            logger.warning(f"Cleanup warning while disconnecting brokers: {exc}")

    def show_menu(self):
        """Render the focused v1 menu."""
        print("\n" + "=" * 60)
        print("AI Gold Research System - Main Menu")
        print("=" * 60)
        print("1. Import local dataset and prepare features")
        print("2. Train AI models")
        print("3. Run backtest")
        print("4. View backtest reports")
        print("5. Analyze model performance")
        print("6. Manage Exness broker profiles")
        print("7. View system status")
        print("8. View system configuration")
        print("0. Exit")
        print("=" * 60)

    def run(self):
        """Run the main operator loop."""
        self.welcome()

        while True:
            try:
                self.show_menu()
                choice = input("\nSelect an option (0-8): ").strip()

                if choice == "0":
                    print("\nThanks for using the AI Gold Research System.")
                    self.cleanup()
                    break
                if choice == "1":
                    self.collect_and_prepare_data()
                elif choice == "2":
                    self.train_models()
                elif choice == "3":
                    self.run_backtest()
                elif choice == "4":
                    self.show_backtest_reports()
                elif choice == "5":
                    self.analyze_model_performance()
                elif choice == "6":
                    self.broker_interface_management()
                elif choice == "7":
                    self.show_system_status()
                elif choice == "8":
                    self.system_configuration()
                else:
                    print("❌ Invalid selection, please try again.")

                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\n⚠️ Operation interrupted by user")
                self.cleanup()
                break
            except Exception as exc:
                logger.error(f"Main loop error: {exc}")
                print(f"❌ Unexpected error: {exc}")
                input("\nPress Enter to continue...")


def main():
    """Application entry point."""
    setup_logging()
    try:
        system = TradingSystem()
        setup_directories(system.config)
        system.run()
    except ConfigValidationError as exc:
        logger.critical(f"System startup failed: {exc}")
        print("System startup failed: invalid configuration. Please check config/config.yaml")
    except Exception as exc:
        logger.critical(f"System startup failed: {exc}")
        print(f"System startup failed: {exc}")


if __name__ == "__main__":
    main()
