import unittest

import numpy as np
import pandas as pd

from src.ai_models import AIModelManager
from src.backtester import Backtester


class DummyScaler:
    def transform(self, frame):
        return frame.to_numpy(dtype=np.float64)


class DummyModel:
    def __init__(self, predictions, probabilities):
        self._predictions = np.asarray(predictions)
        self._probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict(self, features):
        return self._predictions[: len(features)]

    def predict_proba(self, features):
        return self._probabilities[: len(features)]


class StubAIManager:
    def __init__(self, prediction_frame):
        self.prediction_frame = prediction_frame

    def predict_ensemble_batch(self, feature_data, feature_columns=None, method='voting'):
        return self.prediction_frame.reindex(feature_data.index)


class BacktestingRefactorTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ai_model": {
                "type": "ensemble",
                "models": ["random_forest", "xgboost", "logistic_regression"],
            },
            "backtest": {
                "initial_capital": 10000,
                "commission": 0.0,
                "slippage": 0.0,
                "signal_confidence_threshold": 0.8,
            },
            "trading": {
                "symbol": "XAUUSDm",
                "position_size": 0.01,
                "confidence_threshold": 0.6,
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
            },
            "live_trading": {
                "exit_management": {
                    "mode": "disabled",
                },
            },
        }

    def test_predict_ensemble_batch_preserves_alignment_and_invalid_rows(self):
        manager = AIModelManager(self.config)
        manager.feature_columns = ["feat1", "feat2"]
        manager.target_column = "Future_Direction_1"
        manager.scalers[manager.target_column] = DummyScaler()
        manager.models = {
            "random_forest": DummyModel(
                predictions=[1, 0],
                probabilities=[[0.2, 0.8], [0.7, 0.3]],
            ),
            "xgboost": DummyModel(
                predictions=[1, 1],
                probabilities=[[0.1, 0.9], [0.45, 0.55]],
            ),
            "logistic_regression": DummyModel(
                predictions=[0, 1],
                probabilities=[[0.4, 0.6], [0.2, 0.8]],
            ),
        }

        feature_data = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, np.nan],
                "feat2": [3.0, 4.0, 5.0],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="5min"),
        )

        results = manager.predict_ensemble_batch(feature_data)

        self.assertEqual(list(results.index), list(feature_data.index))
        self.assertTrue(bool(results.iloc[0]["is_valid"]))
        self.assertTrue(bool(results.iloc[1]["is_valid"]))
        self.assertFalse(bool(results.iloc[2]["is_valid"]))
        self.assertEqual(int(results.iloc[0]["prediction"]), 1)
        self.assertEqual(int(results.iloc[1]["prediction"]), 1)
        self.assertAlmostEqual(float(results.iloc[0]["confidence"]), (0.8 + 0.9 + 0.6) / 3, places=6)

    def test_run_backtest_uses_prepared_feature_data_directly(self):
        backtester = Backtester(self.config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.0, 101.0, 102.0],
                "Volume": [10.0, 10.0, 10.0],
                "feat1": [0.1, 0.2, 0.3],
                "feat2": [0.4, 0.5, 0.6],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="5min"),
        )
        predictions = pd.DataFrame(
            {
                "is_valid": [True, True, True],
                "prediction": [1.0, 1.0, 1.0],
                "confidence": [0.8, 0.8, 0.8],
            },
            index=prepared_data.index,
        )

        result = backtester.run_backtest(
            prepared_data,
            StubAIManager(predictions),
            ["feat1", "feat2"],
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.winning_trades, 1)
        self.assertGreater(result.total_pnl, 0)

    def test_run_backtest_uses_configured_backtest_confidence_threshold(self):
        backtester = Backtester(self.config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.0, 101.0],
                "Volume": [10.0, 10.0],
                "feat1": [0.1, 0.2],
                "feat2": [0.4, 0.5],
            },
            index=pd.date_range("2025-01-01", periods=2, freq="5min"),
        )
        predictions = pd.DataFrame(
            {
                "is_valid": [True, True],
                "prediction": [1.0, 1.0],
                "confidence": [0.79, 0.79],
            },
            index=prepared_data.index,
        )

        result = backtester.run_backtest(
            prepared_data,
            StubAIManager(predictions),
            ["feat1", "feat2"],
        )

        self.assertEqual(result.total_trades, 0)

    def test_persistence_baseline_predictions_use_only_past_realized_moves(self):
        backtester = Backtester(self.config)
        prepared_data = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 99.0, 102.0],
            },
            index=pd.date_range("2025-01-01", periods=4, freq="5min"),
        )

        baseline_predictions = backtester.build_persistence_baseline_predictions(prepared_data)

        self.assertFalse(bool(baseline_predictions.iloc[0]["is_valid"]))
        self.assertFalse(bool(baseline_predictions.iloc[1]["is_valid"]))
        self.assertTrue(bool(baseline_predictions.iloc[2]["is_valid"]))
        self.assertEqual(float(baseline_predictions.iloc[2]["prediction"]), 1.0)
        self.assertEqual(float(baseline_predictions.iloc[3]["prediction"]), 0.0)

    def test_run_backtest_comparison_returns_curve_data_for_model_and_baseline(self):
        backtester = Backtester(self.config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 100.0, 102.0],
                "High": [101.0, 102.0, 101.0, 103.0],
                "Low": [99.0, 100.0, 99.0, 101.0],
                "Close": [100.0, 101.0, 100.0, 102.0],
                "Volume": [10.0, 10.0, 10.0, 10.0],
                "feat1": [0.1, 0.2, 0.3, 0.4],
                "feat2": [0.4, 0.5, 0.6, 0.7],
            },
            index=pd.date_range("2025-01-01", periods=4, freq="5min"),
        )
        predictions = pd.DataFrame(
            {
                "is_valid": [True, True, True, True],
                "prediction": [1.0, 1.0, 0.0, 1.0],
                "confidence": [0.9, 0.9, 0.9, 0.9],
            },
            index=prepared_data.index,
        )

        comparison = backtester.run_backtest_comparison(
            prepared_data,
            StubAIManager(predictions),
            ["feat1", "feat2"],
        )

        self.assertEqual(comparison.baseline_name, "persistence")
        self.assertIn("model_equity", comparison.equity_curve_data.columns)
        self.assertIn("baseline_equity", comparison.equity_curve_data.columns)
        self.assertIn("model_return_pct", comparison.equity_curve_data.columns)
        self.assertGreater(len(comparison.equity_curve_data), 0)
        self.assertIsNotNone(comparison.model_result.equity_curve)
        self.assertIsNotNone(comparison.baseline_result.equity_curve)

    def test_run_backtest_does_not_reenter_on_same_candle_after_close(self):
        config = {
            "ai_model": {
                "type": "ensemble",
                "models": ["random_forest"],
            },
            "backtest": {
                "initial_capital": 10000,
                "commission": 0.0,
                "slippage": 0.0,
                "signal_confidence_threshold": 0.6,
            },
            "trading": {
                "symbol": "XAUUSDm",
                "position_size": 1.0,
                "confidence_threshold": 0.6,
                "stop_loss_pips": 1,
                "take_profit_pips": 1,
            },
        }
        backtester = Backtester(config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 102.0, 102.0],
                "High": [100.0, 102.0, 102.0],
                "Low": [100.0, 102.0, 102.0],
                "Close": [100.0, 102.0, 102.0],
                "Volume": [10.0, 10.0, 10.0],
                "feat1": [0.1, 0.2, 0.3],
                "feat2": [0.4, 0.5, 0.6],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="5min"),
        )
        predictions = pd.DataFrame(
            {
                "is_valid": [True, True, True],
                "prediction": [1.0, 1.0, 1.0],
                "confidence": [0.8, 0.8, 0.8],
            },
            index=prepared_data.index,
        )

        result = backtester.run_backtest(
            prepared_data,
            StubAIManager(predictions),
            ["feat1", "feat2"],
        )

        closed_trades = [trade for trade in backtester.trades if trade.status == "closed"]
        same_candle_reentries = sum(
            1
            for previous, current in zip(closed_trades, closed_trades[1:])
            if previous.exit_time == current.entry_time
        )
        self.assertEqual(same_candle_reentries, 0)

    def test_run_backtest_requires_selected_features_to_exist(self):
        backtester = Backtester(self.config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [10.0],
            },
            index=pd.date_range("2025-01-01", periods=1, freq="5min"),
        )
        predictions = pd.DataFrame(
            {"is_valid": [True], "prediction": [1.0], "confidence": [0.8]},
            index=prepared_data.index,
        )

        with self.assertRaises(ValueError):
            backtester.run_backtest(prepared_data, StubAIManager(predictions), ["feat1"])

    def test_backtest_trailing_stop_can_close_trade_earlier_than_fixed_stop(self):
        config = {
            "ai_model": {
                "type": "ensemble",
                "models": ["random_forest"],
            },
            "backtest": {
                "initial_capital": 10000,
                "commission": 0.0,
                "slippage": 0.0,
                "signal_confidence_threshold": 0.6,
            },
            "trading": {
                "symbol": "XAUUSDm",
                "position_size": 1.0,
                "confidence_threshold": 0.6,
                "stop_loss_pips": 10.0,
                "take_profit_pips": 100.0,
            },
            "live_trading": {
                "exit_management": {
                    "mode": "trailing_stop",
                    "break_even_enabled": True,
                    "break_even_trigger_pips": 2.0,
                    "break_even_offset_pips": 0.5,
                    "trailing_enabled": True,
                    "trailing_activation_pips": 4.0,
                    "trailing_distance_pips": 1.5,
                    "trailing_step_pips": 0.5,
                },
            },
        }
        backtester = Backtester(config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 103.0],
                "High": [100.0, 105.0, 103.0],
                "Low": [100.0, 105.0, 103.0],
                "Close": [100.0, 105.0, 103.0],
                "Volume": [10.0, 10.0, 10.0],
                "feat1": [0.1, 0.2, 0.3],
                "feat2": [0.4, 0.5, 0.6],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="5min"),
        )
        predictions = pd.DataFrame(
            {
                "is_valid": [True, True, True],
                "prediction": [1.0, 1.0, 1.0],
                "confidence": [0.8, 0.8, 0.8],
            },
            index=prepared_data.index,
        )

        result = backtester.run_backtest(prepared_data, StubAIManager(predictions), ["feat1", "feat2"])

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(backtester.trades[0].reason, "stop_loss")
        self.assertEqual(backtester.trades[0].exit_time, prepared_data.index[2])
        self.assertGreater(backtester.trades[0].stop_loss, backtester.trades[0].entry_price)

    def test_backtest_disabled_exit_management_keeps_fixed_exit_behavior(self):
        config = {
            "ai_model": {
                "type": "ensemble",
                "models": ["random_forest"],
            },
            "backtest": {
                "initial_capital": 10000,
                "commission": 0.0,
                "slippage": 0.0,
                "signal_confidence_threshold": 0.6,
            },
            "trading": {
                "symbol": "XAUUSDm",
                "position_size": 1.0,
                "confidence_threshold": 0.6,
                "stop_loss_pips": 10.0,
                "take_profit_pips": 100.0,
            },
            "live_trading": {
                "exit_management": {
                    "mode": "disabled",
                },
            },
        }
        backtester = Backtester(config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 103.0],
                "High": [100.0, 105.0, 103.0],
                "Low": [100.0, 105.0, 103.0],
                "Close": [100.0, 105.0, 103.0],
                "Volume": [10.0, 10.0, 10.0],
                "feat1": [0.1, 0.2, 0.3],
                "feat2": [0.4, 0.5, 0.6],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="5min"),
        )
        predictions = pd.DataFrame(
            {
                "is_valid": [True, True, True],
                "prediction": [1.0, 1.0, 1.0],
                "confidence": [0.8, 0.8, 0.8],
            },
            index=prepared_data.index,
        )

        result = backtester.run_backtest(prepared_data, StubAIManager(predictions), ["feat1", "feat2"])

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(backtester.trades[0].reason, "backtest_end")
        self.assertEqual(backtester.trades[0].exit_time, prepared_data.index[-1])

    def test_intrabar_buy_stop_loss_hits_on_low_even_when_close_survives(self):
        config = {
            "ai_model": {"type": "ensemble", "models": ["random_forest"]},
            "backtest": {"initial_capital": 10000, "commission": 0.0, "slippage": 0.0, "signal_confidence_threshold": 0.6},
            "trading": {"symbol": "XAUUSDm", "position_size": 1.0, "confidence_threshold": 0.6, "stop_loss_pips": 5.0, "take_profit_pips": 20.0},
            "live_trading": {"exit_management": {"mode": "disabled"}},
        }
        backtester = Backtester(config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 103.0],
                "Low": [100.0, 94.0],
                "Close": [100.0, 101.0],
                "Volume": [10.0, 10.0],
                "feat1": [0.1, 0.2],
                "feat2": [0.4, 0.5],
            },
            index=pd.date_range("2025-01-01", periods=2, freq="5min"),
        )
        predictions = pd.DataFrame(
            {"is_valid": [True, True], "prediction": [1.0, 1.0], "confidence": [0.8, 0.8]},
            index=prepared_data.index,
        )

        result = backtester.run_backtest(prepared_data, StubAIManager(predictions), ["feat1", "feat2"])

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(backtester.trades[0].reason, "stop_loss")
        self.assertEqual(backtester.trades[0].exit_trigger_price, 95.0)

    def test_intrabar_buy_take_profit_hits_on_high_even_when_close_falls_back(self):
        config = {
            "ai_model": {"type": "ensemble", "models": ["random_forest"]},
            "backtest": {"initial_capital": 10000, "commission": 0.0, "slippage": 0.0, "signal_confidence_threshold": 0.6},
            "trading": {"symbol": "XAUUSDm", "position_size": 1.0, "confidence_threshold": 0.6, "stop_loss_pips": 20.0, "take_profit_pips": 5.0},
            "live_trading": {"exit_management": {"mode": "disabled"}},
        }
        backtester = Backtester(config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 106.0],
                "Low": [100.0, 99.0],
                "Close": [100.0, 101.0],
                "Volume": [10.0, 10.0],
                "feat1": [0.1, 0.2],
                "feat2": [0.4, 0.5],
            },
            index=pd.date_range("2025-01-01", periods=2, freq="5min"),
        )
        predictions = pd.DataFrame(
            {"is_valid": [True, True], "prediction": [1.0, 1.0], "confidence": [0.8, 0.8]},
            index=prepared_data.index,
        )

        result = backtester.run_backtest(prepared_data, StubAIManager(predictions), ["feat1", "feat2"])

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(backtester.trades[0].reason, "take_profit")
        self.assertEqual(backtester.trades[0].exit_trigger_price, 105.0)

    def test_same_bar_stop_and_take_profit_uses_conservative_exit(self):
        config = {
            "ai_model": {"type": "ensemble", "models": ["random_forest"]},
            "backtest": {"initial_capital": 10000, "commission": 0.0, "slippage": 0.0, "signal_confidence_threshold": 0.6},
            "trading": {"symbol": "XAUUSDm", "position_size": 1.0, "confidence_threshold": 0.6, "stop_loss_pips": 5.0, "take_profit_pips": 5.0},
            "live_trading": {"exit_management": {"mode": "disabled"}},
        }
        backtester = Backtester(config)
        prepared_data = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 106.0],
                "Low": [100.0, 94.0],
                "Close": [100.0, 100.0],
                "Volume": [10.0, 10.0],
                "feat1": [0.1, 0.2],
                "feat2": [0.4, 0.5],
            },
            index=pd.date_range("2025-01-01", periods=2, freq="5min"),
        )
        predictions = pd.DataFrame(
            {"is_valid": [True, True], "prediction": [1.0, 1.0], "confidence": [0.8, 0.8]},
            index=prepared_data.index,
        )

        result = backtester.run_backtest(prepared_data, StubAIManager(predictions), ["feat1", "feat2"])

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(backtester.trades[0].reason, "stop_loss")
        self.assertEqual(backtester.trades[0].exit_trigger_price, 95.0)


if __name__ == "__main__":
    unittest.main()
