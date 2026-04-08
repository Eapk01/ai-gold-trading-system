"""
Exness demo auto-trader runtime.
"""

from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from src.config_utils import get_effective_confidence_threshold


class LiveDemoTrader:
    """Small Exness-first demo trading runtime for the GUI/service app."""

    def __init__(self, config: Dict[str, Any], broker_manager, feature_engineer, ai_model_manager, runtime_predictor_getter=None):
        self.config = config
        self.broker_manager = broker_manager
        self.feature_engineer = feature_engineer
        self.ai_model_manager = ai_model_manager
        self.runtime_predictor_getter = runtime_predictor_getter

        live_config = config.get("live_trading", {})
        self.symbol = config["trading"]["symbol"]
        self.timeframe = config["trading"]["timeframe"]
        self.position_size = float(config["trading"]["position_size"])
        self.stop_loss_pips = float(config["trading"]["stop_loss_pips"])
        self.take_profit_pips = float(config["trading"]["take_profit_pips"])
        self.lookback_periods = int(config["ai_model"]["lookback_periods"])
        self.confidence_threshold = get_effective_confidence_threshold(config, "live_trading")
        self.poll_interval_seconds = int(live_config.get("poll_interval_seconds", 5))
        self.inactive_poll_interval_seconds = int(live_config.get("inactive_poll_interval_seconds", 30))
        self.startup_candle_buffer = int(live_config.get("startup_candle_buffer", 150))
        self.stale_candle_multiplier = int(live_config.get("stale_candle_multiplier", 4))
        self.history_limit = max(self.startup_candle_buffer * 3, self.lookback_periods + 50, 300)
        self.demo_only = bool(live_config.get("enabled_demo_only", True))
        self.candle_duration_seconds = self._timeframe_to_seconds(self.timeframe)
        self.stale_threshold_seconds = self.candle_duration_seconds * self.stale_candle_multiplier

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._events: Deque[Dict[str, Any]] = deque(maxlen=200)

        self._price_history = pd.DataFrame()
        self._last_processed_candle: Optional[datetime] = None
        self._latest_signal: Optional[Dict[str, Any]] = None
        self._latest_action: str = "idle"
        self._latest_error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._running = False
        self._startup_ready = False
        self._market_state = "idle"
        self._last_candle_age_seconds: Optional[int] = None

    def start(self) -> Tuple[bool, str]:
        """Start the background trading runtime."""
        with self._lock:
            if self._running:
                return True, "Auto trader is already running"

        if self.demo_only and not self._active_broker_is_demo():
            self._append_event(
                "warning",
                "Demo-only mode is enabled, but the active broker profile is not explicitly marked as sandbox/demo",
            )

        if not self.broker_manager.active_broker:
            return False, "Connect an Exness broker before starting the auto trader"

        predictor = self._get_runtime_predictor()
        if predictor is None or not predictor.required_feature_columns:
            return False, "Load a saved model before starting the auto trader"

        success, message = self._initialize_history()
        if not success:
            return False, message

        with self._lock:
            self._stop_event.clear()
            self._running = True
            self._started_at = datetime.now().isoformat()
            self._latest_error = None
            self._thread = threading.Thread(target=self._run_loop, name="live-demo-trader", daemon=True)
            self._thread.start()

        self._append_event("info", "Auto trader started", {"symbol": self.symbol, "timeframe": self.timeframe})
        return True, "Auto trader started"

    def stop(self) -> Tuple[bool, str]:
        """Stop the background trading runtime."""
        with self._lock:
            if not self._running:
                return True, "Auto trader is not running"
            self._running = False
            thread = self._thread
            self._stop_event.set()

        if thread and thread.is_alive():
            thread.join(timeout=max(self.poll_interval_seconds + 1, 2))

        with self._lock:
            self._thread = None

        self._append_event("info", "Auto trader stopped")
        return True, "Auto trader stopped"

    def run_once(self) -> Tuple[bool, str]:
        """Run a single polling iteration. Useful for tests."""
        if not self._startup_ready:
            success, message = self._initialize_history()
            if not success:
                return False, message

        try:
            return self._poll_once()
        except Exception as exc:
            message = f"Auto trader iteration failed: {exc}"
            logger.error(message)
            with self._lock:
                self._latest_error = message
                self._latest_action = "error"
            self._append_event("error", message)
            return False, message

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of runtime state."""
        with self._lock:
            current_positions = self._get_symbol_positions()
            return {
                "running": self._running,
                "startup_ready": self._startup_ready,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "market_state": self._market_state,
                "started_at": self._started_at,
                "last_processed_candle": self._last_processed_candle.isoformat() if self._last_processed_candle else None,
                "last_candle_age_seconds": self._last_candle_age_seconds,
                "latest_signal": deepcopy(self._latest_signal),
                "latest_action": self._latest_action,
                "latest_error": self._latest_error,
                "managed_positions": len(current_positions),
                "loaded_feature_count": len((self._get_runtime_predictor().required_feature_columns if self._get_runtime_predictor() else [])),
                "history_rows": len(self._price_history),
                "confidence_threshold": self.confidence_threshold,
                "active_poll_interval_seconds": self.poll_interval_seconds,
                "inactive_poll_interval_seconds": self.inactive_poll_interval_seconds,
                "stale_threshold_seconds": self.stale_threshold_seconds,
            }

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent event log entries newest first."""
        with self._lock:
            entries = list(self._events)
        return list(reversed(entries[-int(limit) :]))

    def _run_loop(self) -> None:
        while True:
            interval = self._get_current_poll_interval()
            if self._stop_event.wait(interval):
                break
            self.run_once()

    def _initialize_history(self) -> Tuple[bool, str]:
        predictor = self._get_runtime_predictor()
        min_history_rows = predictor.min_history_rows if predictor is not None else 1
        bars_needed = max(self.lookback_periods, min_history_rows, self.startup_candle_buffer)
        candles = self.broker_manager.get_historical_data(self.symbol, self.timeframe, bars_needed + 2)
        if not candles or len(candles) < bars_needed:
            return False, f"Unable to load enough MT5 candle history for {self.symbol} {self.timeframe}"

        closed_candles = candles[:-1] if len(candles) > 1 else candles
        history = self._candles_to_frame(closed_candles)
        if history.empty:
            return False, "Loaded candle history is empty after normalization"

        success, message, _, _ = self._build_latest_signal(history)
        if not success:
            return False, message

        with self._lock:
            candle_age_seconds = self._calculate_candle_age_seconds(history.index[-1].to_pydatetime())
            self._price_history = history.tail(self.history_limit).copy()
            self._last_processed_candle = history.index[-1].to_pydatetime()
            self._startup_ready = True
            self._latest_action = (
                self._format_stale_reason(candle_age_seconds)
                if candle_age_seconds is not None and candle_age_seconds >= self.stale_threshold_seconds
                else "waiting_for_new_candle"
            )
            self._latest_error = None
            self._latest_signal = None
            self._set_market_state_locked(
                "market_closed_or_stale"
                if candle_age_seconds is not None and candle_age_seconds >= self.stale_threshold_seconds
                else "waiting_for_new_candle",
                candle_age_seconds,
            )

        self._append_event(
            "info",
            "Loaded startup candle history",
            {"bars": len(history), "last_closed_candle": self._last_processed_candle.isoformat()},
        )
        return True, "Startup candle history loaded"

    def _poll_once(self) -> Tuple[bool, str]:
        candles = self.broker_manager.get_historical_data(self.symbol, self.timeframe, self.history_limit + 2)
        if not candles or len(candles) < 3:
            return False, "No recent MT5 candles available"

        closed_history = self._candles_to_frame(candles[:-1])
        if closed_history.empty:
            return False, "No valid closed candles available"

        latest_closed = closed_history.index[-1].to_pydatetime()
        with self._lock:
            last_processed = self._last_processed_candle

        if last_processed is not None and latest_closed <= last_processed:
            candle_age_seconds = self._calculate_candle_age_seconds(latest_closed)
            stale_reason = self._format_stale_reason(candle_age_seconds)
            with self._lock:
                self._price_history = closed_history.tail(self.history_limit).copy()
                self._latest_signal = None
                if candle_age_seconds >= self.stale_threshold_seconds:
                    self._latest_action = stale_reason
                    self._set_market_state_locked("market_closed_or_stale", candle_age_seconds)
                else:
                    self._latest_action = "waiting_for_new_candle"
                    self._set_market_state_locked("waiting_for_new_candle", candle_age_seconds)
            return True, stale_reason if candle_age_seconds >= self.stale_threshold_seconds else "No new closed candle"

        success, message, signal, feature_history = self._build_latest_signal(closed_history)
        with self._lock:
            self._price_history = closed_history.tail(self.history_limit).copy()
            self._last_processed_candle = latest_closed
            self._set_market_state_locked("active", self._calculate_candle_age_seconds(latest_closed))

        if not success:
            with self._lock:
                self._latest_signal = None
                self._latest_action = "skipped_invalid_features"
                self._latest_error = message
            self._append_event("warning", message)
            return False, message

        action_message = self._apply_signal(signal)
        with self._lock:
            self._latest_signal = signal
            self._latest_error = None
            self._latest_action = action_message

        self._append_event("info", action_message, {"signal": signal})
        return True, action_message

    def _get_current_poll_interval(self) -> int:
        with self._lock:
            if self._market_state == "market_closed_or_stale":
                return self.inactive_poll_interval_seconds
        return self.poll_interval_seconds

    def _set_market_state_locked(self, new_state: str, candle_age_seconds: Optional[int]) -> None:
        previous_state = self._market_state
        self._market_state = new_state
        self._last_candle_age_seconds = candle_age_seconds

        if previous_state != "market_closed_or_stale" and new_state == "market_closed_or_stale":
            self._append_event_locked(
                "warning",
                "Market appears closed or stale",
                {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "last_candle_age_seconds": candle_age_seconds,
                    "inactive_poll_interval_seconds": self.inactive_poll_interval_seconds,
                    "reason": self._format_stale_reason(candle_age_seconds),
                },
            )
        elif previous_state == "market_closed_or_stale" and new_state != "market_closed_or_stale":
            self._append_event_locked(
                "info",
                "Fresh market data resumed",
                {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "market_state": new_state,
                    "last_candle_age_seconds": candle_age_seconds,
                },
            )

    def _calculate_candle_age_seconds(self, candle_time: Optional[datetime]) -> Optional[int]:
        if candle_time is None:
            return None
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)
        else:
            candle_time = candle_time.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age_seconds = int(max((now_utc - candle_time).total_seconds(), 0))
        return age_seconds

    def _format_stale_reason(self, candle_age_seconds: Optional[int]) -> str:
        if candle_age_seconds is None:
            return f"No new {self.timeframe} candle"
        if candle_age_seconds < self.stale_threshold_seconds:
            return "No new closed candle"
        rounded_minutes = max(round(candle_age_seconds / 60), 1)
        return f"No new {self.timeframe} candle for {rounded_minutes} minutes"

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        normalized = str(timeframe).strip().lower()
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return mapping.get(normalized, 300)

    def _build_latest_signal(
        self, history: pd.DataFrame
    ) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
        feature_history = self.feature_engineer.create_feature_matrix(history, include_targets=False)
        if feature_history.empty:
            return False, "Feature matrix is empty for the current MT5 history", None, None

        predictor = self._get_runtime_predictor()
        if predictor is None:
            return False, "No runtime predictor is loaded", None, feature_history

        selected_features = list(predictor.required_feature_columns)
        missing = [feature for feature in selected_features if feature not in feature_history.columns]
        if missing:
            return False, f"Live feature matrix is missing model features: {missing}", None, feature_history

        prediction_frame = predictor.predict_batch(feature_history)
        latest_prediction = prediction_frame.iloc[-1]
        if not bool(latest_prediction["is_valid"]):
            return False, "Latest live feature row is invalid and cannot be scored", None, feature_history

        prediction = int(latest_prediction["prediction"])
        confidence = float(latest_prediction["confidence"])
        latest_row = feature_history.iloc[-1]
        price = float(pd.to_numeric(latest_row.get("Close"), errors="coerce"))
        signal_side = "buy" if prediction == 1 else "sell"
        stop_loss, take_profit = self._calculate_stop_levels(signal_side, price)
        signal = {
            "timestamp": feature_history.index[-1].isoformat(),
            "prediction": prediction,
            "side": signal_side,
            "confidence": confidence,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        return True, "Live signal generated", signal, feature_history

    def _get_runtime_predictor(self):
        if callable(self.runtime_predictor_getter):
            try:
                return self.runtime_predictor_getter()
            except Exception as exc:
                logger.warning(f"Failed to resolve runtime predictor: {exc}")
                return None
        return None

    def _apply_signal(self, signal: Dict[str, Any]) -> str:
        confidence = float(signal["confidence"])
        if confidence < self.confidence_threshold:
            return f"Holding: confidence {confidence:.3f} is below threshold {self.confidence_threshold:.3f}"

        positions = self._get_symbol_positions()
        if len(positions) > 1:
            return f"Holding: multiple open positions detected for {self.symbol}"

        desired_side = signal["side"]
        if not positions:
            order_result = self._open_position(desired_side, signal)
            if order_result.get("success"):
                return f"Opened {desired_side} position at {signal['price']:.2f}"
            return f"Open order failed: {order_result.get('error', 'unknown error')}"

        current_position = positions[0]
        current_side = self._position_side(current_position)
        if current_side == desired_side:
            return f"Holding existing {current_side} position"

        close_result = self.broker_manager.close_position(str(current_position.get("ticket", "")))
        if not close_result.get("success"):
            return f"Failed to close opposite {current_side} position: {close_result.get('error', 'unknown error')}"

        open_result = self._open_position(desired_side, signal)
        if open_result.get("success"):
            return f"Closed {current_side} position and opened {desired_side} position"
        return f"Closed {current_side} position but failed to open {desired_side}: {open_result.get('error', 'unknown error')}"

    def _open_position(self, side: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        return self.broker_manager.place_order(
            symbol=self.symbol,
            side=side,
            quantity=self.position_size,
            order_type="market",
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
        )

    def _calculate_stop_levels(self, side: str, price: float) -> Tuple[float, float]:
        if side == "buy":
            return price - self.stop_loss_pips, price + self.take_profit_pips
        return price + self.stop_loss_pips, price - self.take_profit_pips

    def _candles_to_frame(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(candles or [])
        if frame.empty:
            return pd.DataFrame()

        frame["Timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", errors="coerce")
        frame = frame.dropna(subset=["Timestamp", "open", "high", "low", "close"])
        if frame.empty:
            return pd.DataFrame()

        frame = frame.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="last")
        frame = frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        frame["Open"] = pd.to_numeric(frame["Open"], errors="coerce")
        frame["High"] = pd.to_numeric(frame["High"], errors="coerce")
        frame["Low"] = pd.to_numeric(frame["Low"], errors="coerce")
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame["Volume"] = pd.to_numeric(frame.get("Volume", 0), errors="coerce").fillna(0.0)
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        frame = frame.set_index("Timestamp")
        frame.index.name = "DateTime"
        frame["Timestamp"] = frame.index
        return frame[["Open", "High", "Low", "Close", "Volume", "Timestamp"]]

    def _get_symbol_positions(self) -> List[Dict[str, Any]]:
        positions = self.broker_manager.get_positions()
        return [position for position in positions if str(position.get("symbol", "")).strip() == self.symbol]

    def _position_side(self, position: Dict[str, Any]) -> str:
        try:
            return "buy" if int(position.get("type", 0)) == 0 else "sell"
        except (TypeError, ValueError):
            return "buy"

    def _active_broker_is_demo(self) -> bool:
        broker = getattr(self.broker_manager, "active_broker", None)
        if broker is None:
            return False
        config = getattr(broker, "config", None)
        return bool(getattr(config, "sandbox", False))

    def _append_event(self, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level.upper(),
            "message": message,
            "details": details or {},
        }
        with self._lock:
            self._events.append(event)

    def _append_event_locked(self, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level.upper(),
            "message": message,
            "details": details or {},
        }
        self._events.append(event)
