"""Helpers for sending live bot snapshots to the external Spacetime dashboard."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


class SpacetimeSnapshotPublisher:
    """Publish bot snapshots through the external TypeScript sender script."""

    def __init__(self, config: Dict[str, Any]):
        settings = dict(config.get("live_trading", {}).get("spacetime_dashboard", {}) or {})
        self.enabled = bool(settings.get("enabled", False))
        self.sender_script = str(settings.get("sender_script", "")).strip()
        self.sender_project_dir = str(settings.get("sender_project_dir", "")).strip()
        self.npx_command = str(settings.get("npx_command", "")).strip()
        self.uri = str(settings.get("uri", "")).strip()
        self.database_name = str(settings.get("database_name", "")).strip()

    def publish_runtime_snapshot(
        self,
        *,
        symbol: str,
        timeframe: str,
        latest_signal: Optional[Dict[str, Any]],
        positions: Optional[List[Dict[str, Any]]],
        account_info: Optional[Dict[str, Any]],
        event_message: str,
    ) -> bool:
        """Build and publish one snapshot from runtime state."""
        if not self.enabled:
            return False

        payload = self.build_payload(
            symbol=symbol,
            timeframe=timeframe,
            latest_signal=latest_signal,
            positions=positions,
            account_info=account_info,
            event_message=event_message,
        )
        return self.publish_payload(payload)

    def build_payload(
        self,
        *,
        symbol: str,
        timeframe: str,
        latest_signal: Optional[Dict[str, Any]],
        positions: Optional[List[Dict[str, Any]]],
        account_info: Optional[Dict[str, Any]],
        event_message: str,
    ) -> Dict[str, Any]:
        """Convert current bot state into the reducer payload shape."""
        resolved_positions = list(positions or [])
        resolved_signal = dict(latest_signal or {})
        active_position = resolved_positions[0] if len(resolved_positions) == 1 else None

        side = "NONE"
        status = "idle"
        position_id: int | str = 1
        entry_price = None
        stop_loss = None
        take_profit = None
        pnl = None

        if active_position is not None:
            raw_ticket = active_position.get("ticket")
            if raw_ticket not in (None, ""):
                position_id = str(raw_ticket)
            side = "BUY" if self._position_side(active_position) == "buy" else "SELL"
            status = "active"
            entry_price = self._safe_float(active_position.get("price_open"))
            stop_loss = self._safe_float(active_position.get("sl"))
            take_profit = self._safe_float(active_position.get("tp"))
            pnl = self._safe_float(active_position.get("profit"))

        signal_price = self._safe_float(resolved_signal.get("price"))
        price = signal_price if signal_price is not None else 0.0
        if active_position is not None:
            price = (
                self._safe_float(active_position.get("price_current"))
                or signal_price
                or entry_price
                or 0.0
            )

        return {
            "position_id": position_id,
            "symbol": self._normalize_symbol(symbol),
            "timeframe": self._normalize_timeframe(timeframe),
            "price": float(price),
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": self._safe_float(resolved_signal.get("confidence")),
            "status": status,
            "equity": float(self._safe_float((account_info or {}).get("equity"), fallback=0.0) or 0.0),
            "pnl": pnl,
            "event_message": str(event_message),
            "event_price": signal_price,
            "event_confidence": self._safe_float(resolved_signal.get("confidence")),
            "timestamp": self._utc_timestamp(),
        }

    def publish_payload(self, payload: Dict[str, Any]) -> bool:
        """Send one JSON payload into the external TypeScript sender."""
        if not self.enabled:
            return False

        if not self.sender_script or not self.sender_project_dir or not self.npx_command:
            logger.warning("Spacetime dashboard publisher is enabled but sender paths are incomplete")
            return False

        env = os.environ.copy()
        env["VITE_SPACETIME_URI"] = self.uri
        env["VITE_SPACETIME_DB_NAME"] = self.database_name
        env["BOT_SNAPSHOT"] = json.dumps(payload)

        try:
            subprocess.run(
                [self.npx_command, "tsx", self.sender_script],
                check=True,
                cwd=self.sender_project_dir,
                env=env,
            )
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning(f"Failed to publish Spacetime dashboard snapshot: {exc}")
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        normalized = str(symbol or "").strip().upper()
        # Strip the common MT5 mini/micro suffix used by the local Exness profile.
        if normalized.endswith("M") and len(normalized) > 4:
            return normalized[:-1]
        return normalized

    def _normalize_timeframe(self, timeframe: str) -> str:
        normalized = str(timeframe or "").strip().lower()
        mapping = {
            "1m": "M1",
            "5m": "M5",
            "15m": "M15",
            "30m": "M30",
            "1h": "H1",
            "4h": "H4",
            "1d": "D1",
        }
        return mapping.get(normalized, str(timeframe or "").strip().upper())

    def _position_side(self, position: Dict[str, Any]) -> str:
        try:
            return "buy" if int(position.get("type", 0)) == 0 else "sell"
        except (TypeError, ValueError):
            return "buy"

    def _safe_float(self, value: Any, fallback: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return fallback
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _utc_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
