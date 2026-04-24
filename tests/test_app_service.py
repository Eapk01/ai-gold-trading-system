import unittest
import sys
import types
from tempfile import TemporaryDirectory
from pathlib import Path

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.app_service import ResearchAppService
from src.broker_interface import BrokerConfig, BrokerManager, BrokerType
from src.config_utils import get_default_config, load_config
from src.secret_store import BrokerSecretStore, LocalSettingsStore
from src.services.research_service import ResearchWorkflowService
from src.services.trading_service import TradingWorkflowService


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


class BrokerManagerReloadTests(unittest.TestCase):
    def test_add_broker_preserves_existing_connected_profile_when_config_is_unchanged(self):
        manager = BrokerManager()
        preserved_config = BrokerConfig(
            broker_type=BrokerType.EXNESS,
            login="123456",
            password="secret",
            server="Exness-MT5Trial16",
            terminal_path="",
            sandbox=False,
            account_id="",
            timeout=30,
            max_retries=3,
        )

        class FakeConnectedBroker:
            def __init__(self, config):
                self.config = config
                self.is_connected = True
                self.disconnect_called = False

            def disconnect(self):
                self.disconnect_called = True
                self.is_connected = False

        existing_broker = FakeConnectedBroker(preserved_config)
        manager.brokers["Main"] = existing_broker
        manager.active_broker = existing_broker

        result = manager.add_broker(
            "Main",
            BrokerConfig(
                broker_type=BrokerType.EXNESS,
                login="123456",
                password="",
                server="Exness-MT5Trial16",
                terminal_path="",
                sandbox=False,
                account_id="",
                timeout=30,
                max_retries=3,
            ),
        )

        self.assertTrue(result)
        self.assertIs(manager.brokers["Main"], existing_broker)
        self.assertIs(manager.active_broker, existing_broker)
        self.assertFalse(existing_broker.disconnect_called)
        self.assertEqual(manager.brokers["Main"].config.password, "secret")


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

    def test_local_settings_store_round_trip_for_broker_profiles(self):
        with TemporaryDirectory() as temp_dir:
            store = LocalSettingsStore(Path(temp_dir) / "local_settings.json")

            store.save_broker_profile(
                "Main",
                {
                    "broker_type": "exness",
                    "login": "123456",
                    "server": "Exness-MT5Trial16",
                },
            )
            store.set_default_broker_profile("Main")

            self.assertIn("Main", store.get_broker_profiles())
            self.assertEqual(store.get_default_broker_profile(), "Main")

            store.delete_broker_profile("Main")
            self.assertEqual(store.get_broker_profiles(), {})
            self.assertEqual(store.get_default_broker_profile(), "")


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
            service.local_settings = LocalSettingsStore(Path(temp_dir) / "local_settings.json")
            service._persist_config = lambda: None

            service._migrate_plaintext_broker_secrets()

            self.assertEqual(service.config["brokers"]["profiles"], {})
            self.assertEqual(service.secret_store.get_password("Main"), "secret")
            self.assertIn("Main", service.local_settings.get_broker_profiles())


class RepoConfigSanitizationTests(unittest.TestCase):
    def test_persist_config_strips_local_broker_state_from_repo_file(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            service = ResearchAppService.__new__(ResearchAppService)
            service.config_path = str(config_path)
            service.config = get_default_config()
            service.config["brokers"]["profiles"] = {
                "Main": {
                    "broker_type": "exness",
                    "login": "123456",
                    "server": "Exness-MT5Trial16",
                    "terminal_path": "",
                    "sandbox": False,
                    "account_id": "",
                    "timeout": 30,
                    "max_retries": 3,
                }
            }
            service.config["brokers"]["default_profile"] = "Main"
            service._sanitize_config_for_repo = ResearchAppService._sanitize_config_for_repo.__get__(service, ResearchAppService)
            service._persist_config = ResearchAppService._persist_config.__get__(service, ResearchAppService)

            service._persist_config()
            loaded = load_config(str(config_path))

            self.assertEqual(loaded["brokers"]["profiles"], {})
            self.assertEqual(loaded["brokers"]["default_profile"], "")


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
                        "app:",
                        "  startup:",
                        "    autoload_latest_model: false",
                        "    autoconnect_broker: false",
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


class RuntimeFeatureSelectionTests(unittest.TestCase):
    def test_get_runtime_prediction_features_prefers_loaded_model_features(self):
        service = ResearchAppService.__new__(ResearchAppService)
        service.selected_features = ["research_feature_a", "research_feature_b"]
        service.ai_model_manager = type(
            "Manager",
            (),
            {
                "models": {"random_forest": object()},
                "feature_columns": ["trained_feature_a", "trained_feature_b"],
            },
        )()

        self.assertEqual(
            service.get_runtime_prediction_features(),
            ["trained_feature_a", "trained_feature_b"],
        )

    def test_get_runtime_prediction_features_falls_back_to_selected_features(self):
        service = ResearchAppService.__new__(ResearchAppService)
        service.selected_features = ["research_feature_a", "research_feature_b"]
        service.ai_model_manager = type("Manager", (), {"models": {}, "feature_columns": []})()

        self.assertEqual(
            service.get_runtime_prediction_features(),
            ["research_feature_a", "research_feature_b"],
        )

    def test_run_model_test_uses_loaded_model_feature_columns(self):
        import pandas as pd

        runtime_columns = ["trained_feature_a", "trained_feature_b"]
        feature_data = pd.DataFrame(
            {
                "trained_feature_a": [1.0, 2.0],
                "trained_feature_b": [3.0, 4.0],
                "research_only_feature": [9.0, 9.0],
                "Future_Direction_1": [0, 1],
            }
        )

        class FakeModelManager:
            def __init__(self):
                self.models = {"random_forest": object()}
                self.feature_columns = list(runtime_columns)
                self.target_column = "Future_Direction_1"
                self.last_feature_columns = []

            def predict_ensemble_batch(self, feature_data, feature_columns=None, method="voting"):
                self.last_feature_columns = list(feature_columns or [])
                return pd.DataFrame(
                    {
                        "is_valid": [True, True],
                        "prediction": [0.0, 1.0],
                        "confidence": [0.8, 0.9],
                    },
                    index=feature_data.index,
                )

        class FakeResult:
            def __init__(self):
                self.summary = {"accuracy": 1.0}
                self.threshold_performance = pd.DataFrame([])
                self.confidence_buckets = pd.DataFrame([])
                self.row_evaluations = pd.DataFrame([])

        class FakeModelTester:
            def evaluate(self, feature_data, prediction_frame, target_series):
                return FakeResult()

        service = ResearchAppService.__new__(ResearchAppService)
        service.feature_data = feature_data
        service.selected_features = ["research_only_feature"]
        service.ai_model_manager = FakeModelManager()
        service.model_tester = FakeModelTester()
        service.loaded_model_path = "models/test.joblib"
        service.latest_model_test_summary = {}
        service.latest_model_test_artifacts = {}
        service._reload_runtime_from_disk = lambda: None
        service.get_runtime_prediction_features = ResearchAppService.get_runtime_prediction_features.__get__(
            service,
            ResearchAppService,
        )
        service.get_target_column = lambda: "Future_Direction_1"
        service._serialize_model_test_result = lambda result: dict(result.summary)
        service._response = lambda success, message, **kwargs: {
            "success": success,
            "message": message,
            **kwargs,
        }

        workflow = ResearchWorkflowService(service)
        result = workflow.run_model_test()

        self.assertTrue(result["success"])
        self.assertEqual(service.ai_model_manager.last_feature_columns, runtime_columns)


class SearchCatalogServiceTests(unittest.TestCase):
    def test_get_search_catalog_delegates_to_research_workflows(self):
        service = ResearchAppService.__new__(ResearchAppService)
        expected = {"success": True, "data": {"candidate_count": 12}}
        service._ensure_workflow_services = lambda: None
        service.research_workflows = type(
            "Workflows",
            (),
            {"get_search_catalog": staticmethod(lambda: expected)},
        )()

        result = ResearchAppService.get_search_catalog(service)

        self.assertIs(result, expected)

    def test_get_search_catalog_passes_search_overrides_to_research_workflows(self):
        service = ResearchAppService.__new__(ResearchAppService)
        captured = {}
        expected = {"success": True, "data": {"candidate_count": 1}}
        service._ensure_workflow_services = lambda: None

        class Workflows:
            @staticmethod
            def get_search_catalog(search_overrides=None):
                captured["search_overrides"] = search_overrides
                return expected

        service.research_workflows = Workflows()

        result = ResearchAppService.get_search_catalog(service, {"trainer_name": "lstm"})

        self.assertIs(result, expected)
        self.assertEqual(captured["search_overrides"], {"trainer_name": "lstm"})

    def test_run_automated_search_passes_search_overrides_to_research_workflows(self):
        service = ResearchAppService.__new__(ResearchAppService)
        captured = {}
        expected = {"success": True}
        service._ensure_workflow_services = lambda: None

        class Workflows:
            @staticmethod
            def run_automated_search(search_name, progress_callback=None, max_workers=None, search_overrides=None):
                captured["search_name"] = search_name
                captured["max_workers"] = max_workers
                captured["search_overrides"] = search_overrides
                return expected

        service.research_workflows = Workflows()

        result = ResearchAppService.run_automated_search(
            service,
            "stage5_gui",
            max_workers=2,
            search_overrides={"preset_names": ["capacity"]},
        )

        self.assertIs(result, expected)
        self.assertEqual(captured["search_name"], "stage5_gui")
        self.assertEqual(captured["max_workers"], 2)
        self.assertEqual(captured["search_overrides"], {"preset_names": ["capacity"]})


class AutoTraderSettingsTests(unittest.TestCase):
    def _make_service(self, *, running: bool = False):
        service = ResearchAppService.__new__(ResearchAppService)
        service.config = {
            "trading": {
                "symbol": "XAUUSDm",
                "timeframe": "5m",
                "position_size": 0.01,
                "stop_loss_pips": 50.0,
                "take_profit_pips": 100.0,
                "confidence_threshold": 0.6,
            },
            "live_trading": {
                "signal_confidence_threshold": 0.6,
                "presets": {},
                "exit_management": {
                    "mode": "trailing_stop",
                    "atr_source": "ATR_14",
                    "initial_stop_loss_mode": "fixed",
                    "initial_take_profit_mode": "fixed",
                    "initial_stop_loss_atr_multiplier": 2.0,
                    "initial_take_profit_atr_multiplier": 4.0,
                    "break_even_enabled": True,
                    "break_even_trigger_mode": "fixed",
                    "break_even_trigger_pips": 8.0,
                    "break_even_trigger_atr_multiplier": 1.0,
                    "break_even_offset_mode": "fixed",
                    "break_even_offset_pips": 1.0,
                    "break_even_offset_atr_multiplier": 0.25,
                    "trailing_enabled": True,
                    "trailing_activation_mode": "fixed",
                    "trailing_activation_pips": 12.0,
                    "trailing_activation_atr_multiplier": 1.5,
                    "trailing_distance_mode": "fixed",
                    "trailing_distance_pips": 5.0,
                    "trailing_distance_atr_multiplier": 1.0,
                    "trailing_step_mode": "fixed",
                    "trailing_step_pips": 2.0,
                    "trailing_step_atr_multiplier": 0.25,
                    "keep_take_profit": True,
                },
            },
        }
        service.auto_trader_session_overrides = {}
        service._response = ResearchAppService._response.__get__(service, ResearchAppService)
        service._deep_merge_dicts = ResearchAppService._deep_merge_dicts.__get__(service, ResearchAppService)
        service._get_effective_runtime_config = ResearchAppService._get_effective_runtime_config.__get__(service, ResearchAppService)
        service._persist_called = 0
        service._build_called = 0
        service._persist_config = lambda: setattr(service, "_persist_called", service._persist_called + 1)
        service._build_runtime_components = lambda: setattr(service, "_build_called", service._build_called + 1)
        service.auto_trader = type("AutoTrader", (), {"get_status": staticmethod(lambda: {"running": running})})()
        return service, TradingWorkflowService(service)

    def test_apply_auto_trader_settings_updates_session_without_persisting(self):
        service, workflow = self._make_service(running=False)
        result = workflow.apply_auto_trader_settings(
            {
                "position_size": 0.1,
                "stop_loss_pips": 20.0,
                "take_profit_pips": 35.0,
                "signal_confidence_threshold": 0.58,
                "exit_management_mode": "trailing_stop",
                "atr_source": "ATR_14",
                "initial_stop_loss_mode": "fixed",
                "initial_take_profit_mode": "fixed",
                "initial_stop_loss_atr_multiplier": 2.0,
                "initial_take_profit_atr_multiplier": 4.0,
                "break_even_enabled": True,
                "break_even_trigger_mode": "fixed",
                "break_even_trigger_pips": 6.0,
                "break_even_trigger_atr_multiplier": 1.0,
                "break_even_offset_mode": "fixed",
                "break_even_offset_pips": 0.5,
                "break_even_offset_atr_multiplier": 0.25,
                "trailing_enabled": True,
                "trailing_activation_mode": "fixed",
                "trailing_activation_pips": 9.0,
                "trailing_activation_atr_multiplier": 1.5,
                "trailing_distance_mode": "fixed",
                "trailing_distance_pips": 3.5,
                "trailing_distance_atr_multiplier": 1.0,
                "trailing_step_mode": "fixed",
                "trailing_step_pips": 1.0,
                "trailing_step_atr_multiplier": 0.25,
                "keep_take_profit": True,
            }
        )

        self.assertTrue(result["success"])
        self.assertEqual(service._persist_called, 0)
        self.assertEqual(service._build_called, 1)
        effective = service._get_effective_runtime_config()
        self.assertEqual(effective["trading"]["position_size"], 0.1)
        self.assertEqual(effective["trading"]["stop_loss_pips"], 20.0)
        self.assertEqual(effective["live_trading"]["exit_management"]["trailing_distance_pips"], 3.5)

    def test_save_auto_trader_settings_as_defaults_persists_values(self):
        service, workflow = self._make_service(running=False)
        result = workflow.save_auto_trader_settings_as_defaults(
            {
                "position_size": 0.2,
                "stop_loss_pips": 15.0,
                "take_profit_pips": 25.0,
                "signal_confidence_threshold": 0.56,
                "exit_management_mode": "trailing_stop",
                "atr_source": "ATR_14",
                "initial_stop_loss_mode": "fixed",
                "initial_take_profit_mode": "fixed",
                "initial_stop_loss_atr_multiplier": 2.0,
                "initial_take_profit_atr_multiplier": 4.0,
                "break_even_enabled": True,
                "break_even_trigger_mode": "fixed",
                "break_even_trigger_pips": 4.0,
                "break_even_trigger_atr_multiplier": 1.0,
                "break_even_offset_mode": "fixed",
                "break_even_offset_pips": 0.3,
                "break_even_offset_atr_multiplier": 0.25,
                "trailing_enabled": True,
                "trailing_activation_mode": "fixed",
                "trailing_activation_pips": 6.0,
                "trailing_activation_atr_multiplier": 1.5,
                "trailing_distance_mode": "fixed",
                "trailing_distance_pips": 2.5,
                "trailing_distance_atr_multiplier": 1.0,
                "trailing_step_mode": "fixed",
                "trailing_step_pips": 0.8,
                "trailing_step_atr_multiplier": 0.25,
                "keep_take_profit": True,
            }
        )

        self.assertTrue(result["success"])
        self.assertEqual(service._persist_called, 1)
        self.assertEqual(service.config["trading"]["position_size"], 0.2)
        self.assertEqual(service.config["trading"]["stop_loss_pips"], 15.0)
        self.assertEqual(service.config["live_trading"]["exit_management"]["trailing_step_pips"], 0.8)

    def test_running_auto_trader_apply_requires_restart(self):
        service, workflow = self._make_service(running=True)
        result = workflow.apply_auto_trader_settings(
            {
                "position_size": 0.1,
                "stop_loss_pips": 20.0,
                "take_profit_pips": 35.0,
                "signal_confidence_threshold": 0.58,
                "exit_management_mode": "trailing_stop",
                "atr_source": "ATR_14",
                "initial_stop_loss_mode": "fixed",
                "initial_take_profit_mode": "fixed",
                "initial_stop_loss_atr_multiplier": 2.0,
                "initial_take_profit_atr_multiplier": 4.0,
                "break_even_enabled": True,
                "break_even_trigger_mode": "fixed",
                "break_even_trigger_pips": 6.0,
                "break_even_trigger_atr_multiplier": 1.0,
                "break_even_offset_mode": "fixed",
                "break_even_offset_pips": 0.5,
                "break_even_offset_atr_multiplier": 0.25,
                "trailing_enabled": True,
                "trailing_activation_mode": "fixed",
                "trailing_activation_pips": 9.0,
                "trailing_activation_atr_multiplier": 1.5,
                "trailing_distance_mode": "fixed",
                "trailing_distance_pips": 3.5,
                "trailing_distance_atr_multiplier": 1.0,
                "trailing_step_mode": "fixed",
                "trailing_step_pips": 1.0,
                "trailing_step_atr_multiplier": 0.25,
                "keep_take_profit": True,
            }
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["data"]["restart_required"])
        self.assertEqual(service._build_called, 0)

    def test_custom_auto_trader_presets_can_be_saved_and_deleted(self):
        service, workflow = self._make_service(running=False)
        values = {
            "position_size": 0.1,
            "stop_loss_pips": 20.0,
            "take_profit_pips": 35.0,
            "signal_confidence_threshold": 0.58,
            "exit_management_mode": "trailing_stop",
            "atr_source": "ATR_14",
            "initial_stop_loss_mode": "fixed",
            "initial_take_profit_mode": "fixed",
            "initial_stop_loss_atr_multiplier": 2.0,
            "initial_take_profit_atr_multiplier": 4.0,
            "break_even_enabled": True,
            "break_even_trigger_mode": "fixed",
            "break_even_trigger_pips": 6.0,
            "break_even_trigger_atr_multiplier": 1.0,
            "break_even_offset_mode": "fixed",
            "break_even_offset_pips": 0.5,
            "break_even_offset_atr_multiplier": 0.25,
            "trailing_enabled": True,
            "trailing_activation_mode": "fixed",
            "trailing_activation_pips": 9.0,
            "trailing_activation_atr_multiplier": 1.5,
            "trailing_distance_mode": "fixed",
            "trailing_distance_pips": 3.5,
            "trailing_distance_atr_multiplier": 1.0,
            "trailing_step_mode": "fixed",
            "trailing_step_pips": 1.0,
            "trailing_step_atr_multiplier": 0.25,
            "keep_take_profit": True,
        }

        save_result = workflow.save_auto_trader_preset("Balanced 5m Custom", values)
        catalog = workflow.get_auto_trader_settings_catalog()
        delete_result = workflow.delete_auto_trader_preset("Balanced_5m_Custom")

        self.assertTrue(save_result["success"])
        self.assertTrue(any(preset["id"] == "Balanced_5m_Custom" for preset in catalog["data"]["custom_presets"]))
        self.assertTrue(delete_result["success"])

    def test_auto_trader_settings_catalog_includes_position_size(self):
        service, workflow = self._make_service(running=False)

        catalog = workflow.get_auto_trader_settings_catalog()

        self.assertTrue(catalog["success"])
        self.assertEqual(catalog["data"]["saved_values"]["position_size"], 0.01)
        self.assertEqual(catalog["data"]["session_values"]["position_size"], 0.01)


if __name__ == "__main__":
    unittest.main()
