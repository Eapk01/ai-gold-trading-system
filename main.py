"""
Focused v1 entry point for dataset import, model training, backtesting, and Exness setup.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.app_service import ResearchAppService
from src.config_utils import ConfigValidationError


def setup_logging() -> None:
    """Configure application logging."""
    logger.add(
        "logs/trading_system_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )


class TradingSystem:
    """CLI shell backed by the shared research service."""

    def __init__(self):
        self.service = ResearchAppService()

    def welcome(self) -> None:
        print("\n" + "=" * 60)
        print("AI Gold Research System")
        print("Local dataset import, model training, backtesting, and Exness setup")
        print("=" * 60)

    def cleanup(self) -> None:
        self.service.cleanup()

    def show_menu(self) -> None:
        print("\n" + "=" * 60)
        print("AI Gold Research System - Main Menu")
        print("=" * 60)
        print("1. Import local dataset and prepare features")
        print("2. Train AI models")
        print("3. Run backtest")
        print("4. View backtest reports")
        print("5. Load saved model")
        print("6. Analyze model performance")
        print("7. Manage Exness broker profiles")
        print("8. View system status")
        print("9. View system configuration")
        print("0. Exit")
        print("=" * 60)

    def collect_and_prepare_data(self) -> None:
        result = self.service.import_and_prepare_data()
        print(f"\n{result['message']}")

        summary = (result.get("data") or {}).get("summary", {})
        if summary:
            print(f"Dataset: {summary.get('path')}")
            print(f"Rows: {summary.get('rows')}")
            print(f"Selected features: {summary.get('selected_features')}")

        if result.get("errors"):
            print("\nData quality warnings:")
            for issue in result["errors"]:
                print(f" - {issue}")

    def train_models(self) -> None:
        requested_name = input("Enter a saved model name [default]: ").strip() or "default"
        result = self.service.train_models(requested_name)
        print(f"\n{result['message']}")

        training_results = (result.get("data") or {}).get("training_results", {})
        if training_results:
            print("\n=== Model Training Results ===")
            for model_name, performance in training_results.items():
                accuracy = performance.get("test_accuracy", 0)
                print(f"{model_name}: test accuracy = {accuracy:.4f}")

    def run_backtest(self) -> None:
        result = self.service.run_backtest()
        print(f"\n{result['message']}")
        if not result["success"]:
            return

        summary = (result.get("data") or {}).get("summary", {})
        if summary:
            print("\n=== Backtest Result Summary ===")
            print(f"Total trades: {summary.get('total_trades', 0)}")
            print(f"Winning trades: {summary.get('winning_trades', 0)}")
            print(f"Losing trades: {summary.get('losing_trades', 0)}")
            print(f"Win rate: {summary.get('win_rate', 0):.1%}")
            print(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
            print(f"Max drawdown: {summary.get('max_drawdown', 0):.2%}")
            print(f"Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
            print(f"Profit factor: {summary.get('profit_factor', 0):.2f}")

        artifacts = result.get("artifacts") or {}
        if artifacts:
            print("\nArtifacts:")
            for label, path in artifacts.items():
                print(f"{label}: {path}")

    def show_backtest_reports(self) -> None:
        reports_result = self.service.list_backtest_reports(limit=5)
        reports = reports_result.get("data") or []
        print("\n=== Backtest Reports ===")
        if not reports:
            print("No backtest report files found")
            return

        print("Available backtest reports:")
        for index, report in enumerate(reports, 1):
            print(f"{index}. {report['timestamp']}")

        choice = input("\nSelect a report number (1-5): ").strip()
        if not choice.isdigit():
            print("Invalid selection")
            return

        report_index = int(choice) - 1
        if not 0 <= report_index < len(reports):
            print("Invalid selection")
            return

        result = self.service.get_backtest_report(reports[report_index]["path"])
        if not result["success"]:
            print(result["message"])
            return

        summary = (result.get("data") or {}).get("summary", {})
        print("\n=== Backtest Report Details ===")
        print(f"Total trades: {summary.get('total_trades', 0)}")
        print(f"Winning trades: {summary.get('winning_trades', 0)}")
        print(f"Losing trades: {summary.get('losing_trades', 0)}")
        print(f"Win rate: {summary.get('win_rate', 0):.1%}")
        print(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
        print(f"Max drawdown: {summary.get('max_drawdown', 0):.2%}")
        print(f"Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"Profit factor: {summary.get('profit_factor', 0):.2f}")

    def load_saved_model(self) -> None:
        result = self.service.list_saved_models()
        saved_models = result.get("data") or []
        if not saved_models:
            print("No saved model files found")
            return

        print("\n=== Saved Models ===")
        for index, model in enumerate(saved_models, 1):
            active_marker = " (loaded)" if model["is_loaded"] else ""
            print(f"{index}. {model['name']} - {model['modified_at']}{active_marker}")

        choice = input("\nSelect a model number to load: ").strip()
        if not choice.isdigit():
            print("Invalid selection")
            return

        model_index = int(choice) - 1
        if not 0 <= model_index < len(saved_models):
            print("Invalid selection")
            return

        load_result = self.service.load_saved_model(saved_models[model_index]["path"])
        print(load_result["message"])

    def analyze_model_performance(self) -> None:
        result = self.service.get_model_analysis()
        print("\n=== Model Performance Analysis ===")
        if not result["success"]:
            print(result["message"])
            return

        summary_rows = (result.get("data") or {}).get("summary", [])
        if summary_rows:
            columns = ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1"]
            print("\nModel performance comparison:")
            for row in summary_rows:
                parts = [f"{column}={row[column]}" for column in columns if column in row]
                print(" | ".join(parts))

        feature_importance = (result.get("data") or {}).get("feature_importance", {})
        print("\n=== Feature Importance Analysis ===")
        for model_name, features in feature_importance.items():
            print(f"\n{model_name} top features:")
            for index, (feature, score) in enumerate(features, 1):
                print(f"{index:2d}. {feature}: {score:.4f}")

    def show_system_status(self) -> None:
        result = self.service.get_system_status()
        status = result.get("data") or {}
        print("\n=== System Status ===")
        last_import_summary = status.get("last_import_summary") or {}
        if last_import_summary:
            print(f"Last dataset: {last_import_summary.get('path')}")
            print(f"Imported rows: {last_import_summary.get('rows', 0)}")
            print(f"Selected features: {last_import_summary.get('selected_features', 0)}")
        else:
            print("Dataset: not imported yet")

        print(f"Models trained: {status.get('models_trained', 0)}")
        print(f"Loaded model file: {status.get('loaded_model_file') or 'None'}")
        print(f"Saved model files: {status.get('saved_model_files', 0)}")
        print(f"Saved broker profiles: {status.get('saved_broker_profiles', 0)}")
        print(f"Active broker: {status.get('active_broker') or 'None'}")

    def broker_interface_management(self) -> None:
        try:
            print("\nExness Broker Management")
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
                    existing_profiles = self.service.config.get("brokers", {}).get("profiles", {})
                    overwrite = False
                    if name in existing_profiles:
                        overwrite = input("A saved profile with this name already exists. Overwrite it? (y/N): ").strip().lower() == "y"
                        if not overwrite:
                            print("Profile save canceled")
                            continue

                    result = self.service.save_broker_profile(
                        name=name,
                        login=input("MT5 Login: ").strip(),
                        password=input("MT5 Password: ").strip(),
                        server=input("MT5 Server: ").strip(),
                        terminal_path=input("MT5 Terminal Path (optional): ").strip(),
                        overwrite=overwrite,
                    )
                    print(result["message"])
                    continue

                if choice == "2":
                    result = self.service.list_broker_profiles()
                    print("\nBroker Status:")
                    for broker in result.get("data") or []:
                        active_label = " (active)" if broker["is_active"] else ""
                        connection_label = "connected" if broker["connected"] else "saved"
                        print(f"{broker['name']}{active_label}: type={broker['type']} status={connection_label}")
                    active_broker = self.service.broker_manager.get_broker_status().get("active_broker")
                    print(f"Active broker: {active_broker or 'None'}")
                    continue

                if choice == "3":
                    brokers = (self.service.list_broker_profiles().get("data") or [])
                    if not brokers:
                        print("No brokers configured")
                        continue

                    print("Available brokers:")
                    for index, broker in enumerate(brokers, 1):
                        print(f"{index}. {broker['name']}")

                    try:
                        broker_index = int(input("Select broker (number): ")) - 1
                    except ValueError:
                        print("Please enter a valid number")
                        continue

                    if not 0 <= broker_index < len(brokers):
                        print("Invalid selection")
                        continue

                    result = self.service.connect_broker(brokers[broker_index]["name"])
                    print(result["message"])
                    continue

                if choice == "4":
                    result = self.service.disconnect_all_brokers()
                    print(result["message"])
                    continue

                if choice == "5":
                    brokers = (self.service.list_broker_profiles().get("data") or [])
                    if not brokers:
                        print("No saved broker profiles found")
                        continue

                    print("Saved broker profiles:")
                    for index, broker in enumerate(brokers, 1):
                        print(f"{index}. {broker['name']}")

                    try:
                        broker_index = int(input("Select broker to delete (number): ")) - 1
                    except ValueError:
                        print("Please enter a valid number")
                        continue

                    if not 0 <= broker_index < len(brokers):
                        print("Invalid selection")
                        continue

                    broker_name = brokers[broker_index]["name"]
                    confirm = input(f"Delete saved profile '{broker_name}'? (y/N): ").strip().lower()
                    if confirm == "y":
                        result = self.service.delete_broker_profile(broker_name)
                        print(result["message"])
                    else:
                        print("Deletion canceled")
                    continue

                print("Invalid selection")
        except Exception as exc:
            print(f"Broker interface management failed: {exc}")
            logger.error(f"Broker interface management error: {exc}")

    def system_configuration(self) -> None:
        result = self.service.get_configuration_summary()
        config = result.get("data") or {}
        print("\n=== System Configuration ===")
        print(f"Trading symbol: {config.get('trading_symbol')}")
        print(f"Timeframe: {config.get('timeframe')}")
        print(f"Primary data source: {config.get('primary_data_source')}")
        print(f"Dataset directory: {config.get('dataset_directory')}")
        print(f"Model type: {config.get('model_type')}")
        print(f"Enabled models: {', '.join(config.get('enabled_models', []))}")
        print(f"Database path: {config.get('database_path')}")

    def run(self) -> None:
        self.welcome()

        while True:
            try:
                self.show_menu()
                choice = input("\nSelect an option (0-9): ").strip()

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
                    self.load_saved_model()
                elif choice == "6":
                    self.analyze_model_performance()
                elif choice == "7":
                    self.broker_interface_management()
                elif choice == "8":
                    self.show_system_status()
                elif choice == "9":
                    self.system_configuration()
                else:
                    print("Invalid selection, please try again.")

                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\nOperation interrupted by user")
                self.cleanup()
                break
            except Exception as exc:
                logger.error(f"Main loop error: {exc}")
                print(f"Unexpected error: {exc}")
                input("\nPress Enter to continue...")


def main() -> None:
    """Application entry point."""
    setup_logging()
    try:
        system = TradingSystem()
        system.run()
    except ConfigValidationError as exc:
        logger.critical(f"System startup failed: {exc}")
        print("System startup failed: invalid configuration. Please check config/config.yaml")
    except Exception as exc:
        logger.critical(f"System startup failed: {exc}")
        print(f"System startup failed: {exc}")


if __name__ == "__main__":
    main()
