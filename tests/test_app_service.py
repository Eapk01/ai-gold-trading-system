import unittest
import sys
import types

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.app_service import ResearchAppService


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


if __name__ == "__main__":
    unittest.main()
