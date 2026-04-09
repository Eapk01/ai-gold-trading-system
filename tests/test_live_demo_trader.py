import unittest
from datetime import datetime, timezone

import pandas as pd

from src.live_demo_trader import LiveDemoTrader


def make_candle(ts, open_price, high_price, low_price, close_price, volume=100):
    return {
        "timestamp": ts,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
    }


class FakeBroker:
    def __init__(self, sandbox=True):
        self.config = type("Config", (), {"sandbox": sandbox})()


class FakeBrokerManager:
    def __init__(self, candles, positions=None, sandbox=True):
        self.active_broker = FakeBroker(sandbox=sandbox)
        self._candles = candles
        self.positions = positions or []
        self.orders = []
        self.closed = []
        self.protection_updates = []
        self.update_result = {"success": True}

    def get_historical_data(self, symbol, timeframe, bars=200):
        return list(self._candles)[-bars:]

    def get_positions(self):
        return list(self.positions)

    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"success": True, "order_id": "demo-order"}

    def close_position(self, position_ticket):
        self.closed.append(position_ticket)
        self.positions = [position for position in self.positions if str(position.get("ticket")) != str(position_ticket)]
        return {"success": True, "position_ticket": str(position_ticket)}

    def update_position_protection(self, position_ticket, *, stop_loss=None, take_profit=None):
        self.protection_updates.append(
            {
                "position_ticket": str(position_ticket),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        )
        if self.update_result.get("success"):
            for position in self.positions:
                if str(position.get("ticket")) == str(position_ticket):
                    if stop_loss is not None:
                        position["sl"] = stop_loss
                    if take_profit is not None:
                        position["tp"] = take_profit
        return dict(self.update_result)


class FakeFeatureEngineer:
    def create_feature_matrix(self, data, include_targets=False):
        frame = data.copy()
        frame["feat1"] = frame["Close"].astype(float)
        frame["ATR_14"] = 2.0
        frame["ATR_7"] = 1.0
        return frame


class FakeAIModelManager:
    def __init__(self, prediction=1.0, confidence=0.8):
        self.models = {"demo": object()}
        self.feature_columns = ["feat1"]
        self.prediction = prediction
        self.confidence = confidence

    def predict_ensemble_batch(self, feature_data, feature_columns=None, method="voting"):
        index = feature_data.index
        return pd.DataFrame(
            {
                "is_valid": [True] * len(index),
                "prediction": [self.prediction] * len(index),
                "confidence": [self.confidence] * len(index),
            },
            index=index,
        )


class LiveDemoTraderTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "trading": {
                "symbol": "XAUUSD",
                "timeframe": "5m",
                "position_size": 0.01,
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "confidence_threshold": 0.6,
            },
            "ai_model": {
                "lookback_periods": 3,
            },
            "live_trading": {
                "enabled_demo_only": True,
                "poll_interval_seconds": 1,
                "inactive_poll_interval_seconds": 10,
                "signal_confidence_threshold": 0.6,
                "startup_candle_buffer": 3,
                "stale_candle_multiplier": 4,
                "exit_management": {
                    "mode": "trailing_stop",
                    "break_even_enabled": True,
                    "break_even_trigger_pips": 1.0,
                    "break_even_offset_pips": 0.25,
                    "trailing_enabled": True,
                    "trailing_activation_pips": 2.0,
                    "trailing_distance_pips": 0.5,
                    "trailing_step_pips": 0.2,
                    "keep_take_profit": True,
                },
            },
        }
        now_ts = int(datetime.now(timezone.utc).timestamp())
        last_open_ts = now_ts - 60
        self.base_candles = [
            make_candle(last_open_ts - 4 * 300, 10, 11, 9, 10.5),
            make_candle(last_open_ts - 3 * 300, 10.5, 11.5, 10, 11),
            make_candle(last_open_ts - 2 * 300, 11, 12, 10.5, 11.5),
            make_candle(last_open_ts - 300, 11.5, 12.5, 11, 12),
            make_candle(last_open_ts, 12, 13, 11.5, 12.5),  # treated as current/open on startup
        ]

    def test_start_requires_active_broker(self):
        broker_manager = FakeBrokerManager(self.base_candles)
        broker_manager.active_broker = None
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager())

        success, message = trader.start()

        self.assertFalse(success)
        self.assertIn("Connect an Exness broker", message)

    def test_start_requires_loaded_model(self):
        manager = FakeAIModelManager()
        manager.models = {}
        trader = LiveDemoTrader(self.config, FakeBrokerManager(self.base_candles), FakeFeatureEngineer(), manager)

        success, message = trader.start()

        self.assertFalse(success)
        self.assertIn("Load a saved model", message)

    def test_start_loads_history_and_waits_for_new_candle(self):
        trader = LiveDemoTrader(self.config, FakeBrokerManager(self.base_candles), FakeFeatureEngineer(), FakeAIModelManager())

        success, message = trader.start()
        stop_success, _ = trader.stop()
        status = trader.get_status()

        self.assertTrue(success)
        self.assertTrue(stop_success)
        self.assertIn("started", message.lower())
        self.assertTrue(status["startup_ready"])
        self.assertEqual(status["latest_action"], "waiting_for_new_candle")
        self.assertEqual(status["market_state"], "waiting_for_new_candle")

    def test_run_once_closes_and_reverses_on_opposite_signal(self):
        positions = [{"ticket": "77", "symbol": "XAUUSD", "type": 1, "volume": 0.01, "price_open": 11.0}]
        broker_manager = FakeBrokerManager(self.base_candles, positions=positions)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.5, 13.0, 12.0, 12.8)]

        result, message = trader.run_once()

        self.assertTrue(result)
        self.assertIn("opened buy", message.lower())
        self.assertEqual(broker_manager.closed, ["77"])
        self.assertEqual(len(broker_manager.orders), 1)
        self.assertEqual(broker_manager.orders[0]["side"], "buy")

    def test_run_once_holds_on_low_confidence(self):
        broker_manager = FakeBrokerManager(self.base_candles)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.4))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.5, 13.0, 12.0, 12.8)]

        result, message = trader.run_once()

        self.assertTrue(result)
        self.assertIn("below threshold", message.lower())
        self.assertEqual(len(broker_manager.orders), 0)

    def test_run_once_holds_same_side_position(self):
        positions = [{"ticket": "88", "symbol": "XAUUSD", "type": 0, "volume": 0.01, "price_open": 11.0, "price_current": 12.8, "sl": 10.0, "tp": 13.5}]
        broker_manager = FakeBrokerManager(self.base_candles, positions=positions)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.5, 13.0, 12.0, 12.8)]

        result, message = trader.run_once()

        self.assertTrue(result)
        self.assertIn("holding existing buy position", message.lower())
        self.assertEqual(len(broker_manager.orders), 0)
        self.assertEqual(len(broker_manager.closed), 0)
        self.assertEqual(len(broker_manager.protection_updates), 1)

    def test_run_once_moves_buy_stop_to_break_even_or_better(self):
        positions = [{"ticket": "88", "symbol": "XAUUSD", "type": 0, "volume": 0.01, "price_open": 11.0, "price_current": 12.8, "sl": 10.0, "tp": 14.0}]
        broker_manager = FakeBrokerManager(self.base_candles, positions=positions)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.5, 13.0, 12.0, 12.8)]

        result, message = trader.run_once()
        status = trader.get_status()

        self.assertTrue(result)
        self.assertIn("holding existing buy position", message.lower())
        self.assertEqual(len(broker_manager.protection_updates), 1)
        self.assertGreater(broker_manager.protection_updates[0]["stop_loss"], 11.0)
        self.assertIn("stop", status["last_protection_action"].lower())
        self.assertAlmostEqual(status["last_managed_stop_loss"], broker_manager.protection_updates[0]["stop_loss"])

    def test_run_once_moves_sell_stop_down_when_in_profit(self):
        positions = [{"ticket": "99", "symbol": "XAUUSD", "type": 1, "volume": 0.01, "price_open": 13.5, "price_current": 12.2, "sl": 14.5, "tp": 11.0}]
        broker_manager = FakeBrokerManager(self.base_candles, positions=positions)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=0.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.5, 13.0, 12.0, 12.2)]

        result, message = trader.run_once()

        self.assertTrue(result)
        self.assertIn("holding existing sell position", message.lower())
        self.assertEqual(len(broker_manager.protection_updates), 1)
        self.assertLess(broker_manager.protection_updates[0]["stop_loss"], 13.5)

    def test_run_once_skips_trailing_update_below_step(self):
        positions = [{"ticket": "88", "symbol": "XAUUSD", "type": 0, "volume": 0.01, "price_open": 11.0, "price_current": 13.1, "sl": 12.45, "tp": 14.0}]
        broker_manager = FakeBrokerManager(self.base_candles, positions=positions)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.8, 13.2, 12.4, 13.1)]

        result, _ = trader.run_once()

        self.assertTrue(result)
        self.assertEqual(len(broker_manager.protection_updates), 0)
        self.assertIn("below trailing step", trader.get_status()["last_protection_action"].lower())

    def test_run_once_skips_protection_with_multiple_positions(self):
        positions = [
            {"ticket": "88", "symbol": "XAUUSD", "type": 0, "volume": 0.01, "price_open": 11.0, "price_current": 12.8, "sl": 10.0, "tp": 14.0},
            {"ticket": "89", "symbol": "XAUUSD", "type": 0, "volume": 0.01, "price_open": 11.1, "price_current": 12.8, "sl": 10.1, "tp": 14.1},
        ]
        broker_manager = FakeBrokerManager(self.base_candles, positions=positions)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        broker_manager._candles = self.base_candles + [make_candle(600, 12.5, 13.0, 12.0, 12.8)]

        result, message = trader.run_once()

        self.assertTrue(result)
        self.assertIn("multiple open positions", message.lower())
        self.assertEqual(len(broker_manager.protection_updates), 0)

    def test_stale_history_sets_market_closed_state(self):
        old_now = int(datetime.now(timezone.utc).timestamp()) - 7200
        stale_candles = [
            make_candle(old_now - 4 * 300, 10, 11, 9, 10.5),
            make_candle(old_now - 3 * 300, 10.5, 11.5, 10, 11),
            make_candle(old_now - 2 * 300, 11, 12, 10.5, 11.5),
            make_candle(old_now - 300, 11.5, 12.5, 11, 12),
            make_candle(old_now, 12, 13, 11.5, 12.5),
        ]
        trader = LiveDemoTrader(self.config, FakeBrokerManager(stale_candles), FakeFeatureEngineer(), FakeAIModelManager())

        success, _ = trader._initialize_history()
        status = trader.get_status()

        self.assertTrue(success)
        self.assertEqual(status["market_state"], "market_closed_or_stale")
        self.assertGreaterEqual(status["last_candle_age_seconds"], status["stale_threshold_seconds"])
        self.assertEqual(status["inactive_poll_interval_seconds"], 10)
        self.assertIn("No new 5m candle", status["latest_action"])

    def test_stale_market_recovers_when_fresh_candle_arrives(self):
        old_now = int(datetime.now(timezone.utc).timestamp()) - 7200
        stale_candles = [
            make_candle(old_now - 4 * 300, 10, 11, 9, 10.5),
            make_candle(old_now - 3 * 300, 10.5, 11.5, 10, 11),
            make_candle(old_now - 2 * 300, 11, 12, 10.5, 11.5),
            make_candle(old_now - 300, 11.5, 12.5, 11, 12),
            make_candle(old_now, 12, 13, 11.5, 12.5),
        ]
        broker_manager = FakeBrokerManager(stale_candles)
        trader = LiveDemoTrader(self.config, broker_manager, FakeFeatureEngineer(), FakeAIModelManager(prediction=1.0, confidence=0.9))

        success, _ = trader._initialize_history()
        self.assertTrue(success)
        self.assertEqual(trader.get_status()["market_state"], "market_closed_or_stale")
        self.assertEqual(trader._get_current_poll_interval(), 10)

        fresh_now = int(datetime.now(timezone.utc).timestamp())
        broker_manager._candles = [
            make_candle(fresh_now - 4 * 300, 10, 11, 9, 10.5),
            make_candle(fresh_now - 3 * 300, 10.5, 11.5, 10, 11),
            make_candle(fresh_now - 2 * 300, 11, 12, 10.5, 11.5),
            make_candle(fresh_now - 300, 11.5, 12.5, 11, 12),
            make_candle(fresh_now, 12, 13, 11.5, 12.5),
            make_candle(fresh_now + 300, 12.5, 13.0, 12.0, 12.8),
        ]

        result, _ = trader.run_once()
        status = trader.get_status()
        event_messages = [event["message"] for event in trader.get_recent_events(10)]

        self.assertTrue(result)
        self.assertNotEqual(status["market_state"], "market_closed_or_stale")
        self.assertEqual(trader._get_current_poll_interval(), 1)
        self.assertIn("Fresh market data resumed", event_messages)


if __name__ == "__main__":
    unittest.main()
