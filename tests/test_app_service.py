import unittest
import sys
import types
from tempfile import TemporaryDirectory
from pathlib import Path

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.app_service import ResearchAppService
from src.secret_store import BrokerSecretStore


class DashboardSnapshotTests(unittest.TestCase):
    def test_get_dashboard_snapshot_returns_safe_defaults_without_broker(self):
        service = ResearchAppService.__new__(ResearchAppService)
        service.get_system_status = lambda: {
            "data": {
                "active_broker": None,
                "loaded_model_file": None,
                "saved_model_files": 0,
                "selected_features": 0,
                "last_import_summary": {},
                "latest_backtest_summary": {},
                "latest_backtest_artifacts": {},
                "auto_trader": {
                    "running": False,
                    "market_state": "idle",
                    "latest_action": "idle",
                    "last_processed_candle": None,
                    "managed_positions": 0,
                    "last_candle_age_seconds": None,
                },
            }
        }
        service.get_configuration_summary = lambda: {"data": {"trading_symbol": "XAUUSDm", "timeframe": "5m"}}
        service.get_trading_snapshot = lambda: {"data": {"account": {}, "positions": [], "broker_connected": False}}

        result = service.get_dashboard_snapshot()
        snapshot = result["data"]

        self.assertTrue(result["success"])
        self.assertFalse(snapshot["broker"]["broker_connected"])
        self.assertEqual(snapshot["positions"]["open_positions_count"], 0)
        self.assertEqual(snapshot["positions"]["open_positions_profit_total"], 0.0)
        self.assertFalse(snapshot["research"]["dataset_imported"])

    def test_get_dashboard_snapshot_aggregates_positions_and_auto_trader(self):
        service = ResearchAppService.__new__(ResearchAppService)
        service.get_system_status = lambda: {
            "data": {
                "active_broker": "Main",
                "loaded_model_file": "baseline.joblib",
                "saved_model_files": 3,
                "selected_features": 30,
                "last_import_summary": {"rows": 1000, "feature_rows": 950},
                "latest_backtest_summary": {"total_trades": 12, "total_pnl": 42.5},
                "latest_backtest_artifacts": {"report_file": "reports/test.json"},
                "auto_trader": {
                    "running": True,
                    "market_state": "market_closed_or_stale",
                    "latest_action": "No new 5m candle for 40 minutes",
                    "last_processed_candle": "2026-04-03T12:00:00",
                    "managed_positions": 2,
                    "last_candle_age_seconds": 2400,
                },
            }
        }
        service.get_configuration_summary = lambda: {"data": {"trading_symbol": "XAUUSDm", "timeframe": "5m"}}
        service.get_trading_snapshot = lambda: {
            "data": {
                "account": {"balance": 1000, "equity": 1012.5, "margin_free": 900, "leverage": 2000},
                "positions": [
                    {"ticket": 1, "symbol": "XAUUSDm", "type": 0, "volume": 0.01, "profit": 10.0},
                    {"ticket": 2, "symbol": "XAUUSDm", "type": 1, "volume": 0.02, "profit": -3.5},
                ],
                "broker_connected": True,
            }
        }

        result = service.get_dashboard_snapshot()
        snapshot = result["data"]

        self.assertTrue(result["success"])
        self.assertEqual(snapshot["broker"]["active_broker"], "Main")
        self.assertEqual(snapshot["positions"]["open_positions_count"], 2)
        self.assertEqual(snapshot["positions"]["open_buy_positions"], 1)
        self.assertEqual(snapshot["positions"]["open_sell_positions"], 1)
        self.assertAlmostEqual(snapshot["positions"]["open_positions_profit_total"], 6.5)
        self.assertTrue(snapshot["live_trading"]["running"])
        self.assertEqual(snapshot["live_trading"]["market_state"], "market_closed_or_stale")
        self.assertEqual(snapshot["research"]["loaded_model_file"], "baseline.joblib")


class BrokerAutoconnectTests(unittest.TestCase):
    def test_autoconnect_prefers_default_profile(self):
        class FakeBrokerManager:
            def __init__(self):
                self.attempts = []

            def connect_broker(self, name, password=""):
                self.attempts.append((name, password))
                return name == "Main"

        service = ResearchAppService.__new__(ResearchAppService)
        service.config = {
            "brokers": {
                "profiles": {
                    "Backup": {},
                    "Main": {},
                },
                "default_profile": "Main",
            }
        }
        service.broker_manager = FakeBrokerManager()
        service.secret_store = type("Secrets", (), {"get_password": staticmethod(lambda name: "secret")})()
        service._persist_config = lambda: None

        service._autoconnect_saved_broker()

        self.assertEqual(service.broker_manager.attempts, [("Main", "secret")])
        self.assertEqual(service.config["brokers"]["default_profile"], "Main")

    def test_autoconnect_falls_back_to_first_saved_profile(self):
        class FakeBrokerManager:
            def __init__(self):
                self.attempts = []

            def connect_broker(self, name, password=""):
                self.attempts.append((name, password))
                return name == "Backup"

        service = ResearchAppService.__new__(ResearchAppService)
        service.config = {
            "brokers": {
                "profiles": {
                    "Backup": {},
                    "Second": {},
                },
                "default_profile": "",
            }
        }
        service.broker_manager = FakeBrokerManager()
        service.secret_store = type("Secrets", (), {"get_password": staticmethod(lambda name: "secret")})()
        service._persist_config = lambda: None

        service._autoconnect_saved_broker()

        self.assertEqual(service.broker_manager.attempts, [("Backup", "secret")])
        self.assertEqual(service.config["brokers"]["default_profile"], "Backup")

    def test_autoconnect_skips_profile_when_secret_missing(self):
        class FakeBrokerManager:
            def __init__(self):
                self.attempts = []

            def connect_broker(self, name, password=""):
                self.attempts.append((name, password))
                return False

        service = ResearchAppService.__new__(ResearchAppService)
        service.config = {
            "brokers": {
                "profiles": {
                    "Main": {},
                    "Backup": {},
                },
                "default_profile": "Main",
            }
        }
        service.broker_manager = FakeBrokerManager()
        service.secret_store = type(
            "Secrets",
            (),
            {"get_password": staticmethod(lambda name: "" if name == "Main" else "secret")},
        )()
        service._persist_config = lambda: None

        service._autoconnect_saved_broker()

        self.assertEqual(service.broker_manager.attempts, [("Backup", "secret")])


class SecretStoreTests(unittest.TestCase):
    def test_secret_store_round_trip_and_delete(self):
        with TemporaryDirectory() as temp_dir:
            store = BrokerSecretStore(Path(temp_dir) / "broker_secrets.json")

            store.set_password("Main", "secret")
            self.assertTrue(store.has_password("Main"))
            self.assertEqual(store.get_password("Main"), "secret")

            store.delete_password("Main")
            self.assertFalse(store.has_password("Main"))
            self.assertEqual(store.get_password("Main"), "")


class SecretMigrationTests(unittest.TestCase):
    def test_plaintext_profile_passwords_are_migrated_out_of_config(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "trading:",
                        "  symbol: XAUUSDm",
                        "  timeframe: 5m",
                        "  position_size: 0.01",
                        "  stop_loss_pips: 50",
                        "  take_profit_pips: 100",
                        "data_sources:",
                        "  primary: local_csv",
                        "  dataset_directory: data/imports",
                        "  min_rows: 100",
                        "ai_model:",
                        "  type: ensemble",
                        "  models: [random_forest]",
                        "  lookback_periods: 100",
                        "risk_management:",
                        "  max_daily_loss: 30",
                        "  max_positions: 3",
                        "  risk_per_trade: 0.02",
                        "  drawdown_limit: 0.15",
                        "backtest:",
                        "  initial_capital: 10000",
                        "  commission: 0.0001",
                        "  slippage: 0.0002",
                        "database:",
                        "  path: data/test.db",
                        "logging:",
                        "  level: INFO",
                        "  file_path: logs/test.log",
                        "brokers:",
                        "  profiles:",
                        "    Main:",
                        "      broker_type: exness",
                        "      login: '123456'",
                        "      password: secret",
                        "      server: Exness-MT5Trial16",
                        "      terminal_path: ''",
                        "      sandbox: false",
                        "      account_id: ''",
                        "      timeout: 30",
                        "      max_retries: 3",
                        "  default_profile: Main",
                    ]
                ),
                encoding="utf-8",
            )

            service = ResearchAppService.__new__(ResearchAppService)
            service.config_path = str(config_path)
            service.config = {
                "brokers": {
                    "profiles": {
                        "Main": {
                            "password": "secret",
                            "login": "123456",
                            "server": "Exness-MT5Trial16",
                        }
                    }
                }
            }
            service.secret_store = BrokerSecretStore(Path(temp_dir) / "broker_secrets.json")
            service._persist_config = lambda: None

            service._migrate_plaintext_broker_secrets()

            self.assertEqual(service.config["brokers"]["profiles"]["Main"]["password"], "")
            self.assertEqual(service.secret_store.get_password("Main"), "secret")


class RuntimeReloadTests(unittest.TestCase):
    def test_reload_runtime_rebuilds_all_config_dependent_components(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "trading:",
                        "  symbol: XAUUSDm",
                        "  timeframe: 5m",
                        "  position_size: 0.01",
                        "  stop_loss_pips: 50",
                        "  take_profit_pips: 100",
                        "  confidence_threshold: 0.61",
                        "data_sources:",
                        f"  dataset_directory: {str(temp_path / 'imports').replace(chr(92), '/')}",
                        "  primary: local_csv",
                        "  min_rows: 100",
                        "ai_model:",
                        "  type: ensemble",
                        "  models: [random_forest]",
                        "  lookback_periods: 100",
                        "  target_column: Future_Direction_1",
                        "risk_management:",
                        "  max_daily_loss: 30",
                        "  max_positions: 3",
                        "  risk_per_trade: 0.02",
                        "  drawdown_limit: 0.15",
                        "backtest:",
                        "  initial_capital: 10000",
                        "  commission: 0.0001",
                        "  slippage: 0.0002",
                        "  signal_confidence_threshold: 0.68",
                        "database:",
                        f"  path: {str(temp_path / 'data' / 'test.db').replace(chr(92), '/')}",
                        "logging:",
                        "  level: INFO",
                        f"  file_path: {str(temp_path / 'logs' / 'test.log').replace(chr(92), '/')}",
                        "brokers:",
                        "  profiles: {}",
                        "  default_profile: ''",
                        "live_trading:",
                        "  enabled_demo_only: true",
                        "  poll_interval_seconds: 5",
                        "  inactive_poll_interval_seconds: 30",
                        "  signal_confidence_threshold: 0.73",
                        "  startup_candle_buffer: 150",
                        "  stale_candle_multiplier: 4",
                    ]
                ),
                encoding="utf-8",
            )

            service = ResearchAppService(str(config_path))
            feature_data = object()
            service.feature_data = feature_data
            service.selected_features = ["feat1"]

            config_path.write_text(config_path.read_text(encoding="utf-8").replace("0.68", "0.75").replace("0.73", "0.79"))
            service._reload_runtime_from_disk()

            self.assertIs(service.feature_data, feature_data)
            self.assertEqual(service.selected_features, ["feat1"])
            self.assertAlmostEqual(service.backtester.signal_confidence_threshold, 0.75)
            self.assertAlmostEqual(service.auto_trader.confidence_threshold, 0.79)
            self.assertEqual(service.ai_model_manager.target_column, "Future_Direction_1")


if __name__ == "__main__":
    unittest.main()
