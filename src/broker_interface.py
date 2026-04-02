"""
Exness broker integration module.
Focused v1 adapter built around a local MetaTrader5 terminal session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from importlib import import_module
from typing import Any, Dict, List, Optional

from loguru import logger


class BrokerType(Enum):
    """Supported broker types for the focused v1 product."""

    EXNESS = "exness"


@dataclass
class BrokerConfig:
    """Persisted Exness broker configuration."""

    broker_type: BrokerType
    login: str = ""
    password: str = ""
    server: str = ""
    terminal_path: str = ""
    sandbox: bool = False
    account_id: str = ""
    timeout: int = 30
    max_retries: int = 3


def _import_metatrader5():
    """Import MetaTrader5 lazily so the dependency remains optional."""
    try:
        return import_module("MetaTrader5")
    except ImportError:
        return None


class ExnessBroker:
    """Exness broker adapter using the MetaTrader5 terminal as the transport."""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.is_connected = False
        self.last_heartbeat = datetime.now()
        self.mt5 = None

    def connect(self) -> bool:
        """Connect to Exness through a local MetaTrader5 terminal."""
        try:
            self.mt5 = _import_metatrader5()
            if self.mt5 is None:
                logger.error("MetaTrader5 package is not installed; Exness broker is unavailable")
                return False

            initialize_kwargs = {}
            if self.config.terminal_path:
                initialize_kwargs["path"] = self.config.terminal_path

            if not self.mt5.initialize(**initialize_kwargs):
                logger.error(f"Exness MT5 initialize failed: {self.mt5.last_error()}")
                return False

            login_value = self.config.login or self.config.account_id
            if login_value:
                authorized = self.mt5.login(
                    login=int(login_value),
                    password=self.config.password,
                    server=self.config.server,
                )
                if not authorized:
                    logger.error(f"Exness login failed: {self.mt5.last_error()}")
                    self.mt5.shutdown()
                    return False

            self.is_connected = True
            self.last_heartbeat = datetime.now()
            logger.info("Exness connection established through MetaTrader5")
            return True
        except Exception as exc:
            logger.error(f"Failed to connect Exness broker: {exc}")
            return False

    def disconnect(self):
        """Disconnect the current MetaTrader5 session."""
        try:
            if self.mt5:
                self.mt5.shutdown()
            self.is_connected = False
            logger.info("Exness connection closed")
        except Exception as exc:
            logger.error(f"Failed to disconnect Exness broker: {exc}")

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str, price: Optional[float] = None) -> Dict:
        """Place an order through the MetaTrader5 trading API."""
        try:
            if not self._ensure_connection():
                return {"success": False, "error": "Exness broker is not connected"}

            if not self.mt5.symbol_select(symbol, True):
                return {"success": False, "error": f"Symbol unavailable: {symbol}"}

            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": f"No market tick available for {symbol}"}

            normalized_type = order_type.lower()
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(quantity),
                "deviation": 20,
                "magic": 20260402,
                "comment": "ai-gold-trading-system",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": getattr(self.mt5, "ORDER_FILLING_IOC", 1),
            }

            if normalized_type == "market":
                if side.lower() == "buy":
                    request["type"] = self.mt5.ORDER_TYPE_BUY
                    request["price"] = tick.ask
                else:
                    request["type"] = self.mt5.ORDER_TYPE_SELL
                    request["price"] = tick.bid
            else:
                pending_map = {
                    ("buy", "limit"): self.mt5.ORDER_TYPE_BUY_LIMIT,
                    ("sell", "limit"): self.mt5.ORDER_TYPE_SELL_LIMIT,
                    ("buy", "stop"): self.mt5.ORDER_TYPE_BUY_STOP,
                    ("sell", "stop"): self.mt5.ORDER_TYPE_SELL_STOP,
                }
                pending_type = pending_map.get((side.lower(), normalized_type))
                if pending_type is None:
                    return {"success": False, "error": f"Unsupported Exness order type: {order_type}"}
                request["action"] = self.mt5.TRADE_ACTION_PENDING
                request["type"] = pending_type
                request["price"] = float(price) if price is not None else (tick.ask if side.lower() == "buy" else tick.bid)

            result = self.mt5.order_send(request)
            if result is None:
                return {"success": False, "error": f"order_send returned None: {self.mt5.last_error()}"}

            payload = result._asdict() if hasattr(result, "_asdict") else {"retcode": getattr(result, "retcode", None)}
            success = result.retcode in {
                getattr(self.mt5, "TRADE_RETCODE_DONE", -1),
                getattr(self.mt5, "TRADE_RETCODE_PLACED", -2),
            }
            if success:
                self.last_heartbeat = datetime.now()
                order_id = str(payload.get("order") or payload.get("deal") or payload.get("request_id") or "")
                logger.info(f"Exness order submitted successfully: {order_id}")
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": "filled" if request["action"] == self.mt5.TRADE_ACTION_DEAL else "placed",
                    "data": payload,
                }

            error_msg = payload.get("comment") or f"retcode={payload.get('retcode')}"
            logger.error(f"Exness order failed: {error_msg}")
            return {"success": False, "error": error_msg, "data": payload}
        except Exception as exc:
            logger.error(f"Exness order placement failed: {exc}")
            return {"success": False, "error": str(exc)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending Exness order."""
        try:
            if not self._ensure_connection():
                return False

            request = {
                "action": self.mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            result = self.mt5.order_send(request)
            if result is None:
                return False
            return result.retcode == getattr(self.mt5, "TRADE_RETCODE_DONE", result.retcode)
        except Exception as exc:
            logger.error(f"Exness cancel order failed: {exc}")
            return False

    def get_order_status(self, order_id: str) -> Dict:
        """Get status for a pending or historical order."""
        try:
            if not self._ensure_connection():
                return {}

            orders = self.mt5.orders_get(ticket=int(order_id))
            if orders:
                order = orders[0]
                return order._asdict() if hasattr(order, "_asdict") else {}

            history = self.mt5.history_orders_get(ticket=int(order_id))
            if history:
                order = history[0]
                return order._asdict() if hasattr(order, "_asdict") else {}

            return {}
        except Exception as exc:
            logger.error(f"Exness get order status failed: {exc}")
            return {}

    def get_positions(self) -> List[Dict]:
        """Return open positions from MetaTrader5."""
        try:
            if not self._ensure_connection():
                return []

            positions = self.mt5.positions_get()
            if not positions:
                return []
            return [position._asdict() if hasattr(position, "_asdict") else {} for position in positions]
        except Exception as exc:
            logger.error(f"Exness get positions failed: {exc}")
            return []

    def get_account_info(self) -> Dict:
        """Return connected account information."""
        try:
            if not self._ensure_connection():
                return {}

            account_info = self.mt5.account_info()
            if account_info is None:
                return {}
            return account_info._asdict() if hasattr(account_info, "_asdict") else {}
        except Exception as exc:
            logger.error(f"Exness get account info failed: {exc}")
            return {}

    def get_market_data(self, symbol: str) -> Dict:
        """Return the latest symbol tick data."""
        try:
            if not self._ensure_connection():
                return {}

            if not self.mt5.symbol_select(symbol, True):
                return {}

            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}

            return {
                "symbol": symbol,
                "bid": float(getattr(tick, "bid", 0.0)),
                "ask": float(getattr(tick, "ask", 0.0)),
                "last": float(getattr(tick, "last", 0.0)),
                "timestamp": getattr(tick, "time", 0),
            }
        except Exception as exc:
            logger.error(f"Exness get market data failed: {exc}")
            return {}

    def _ensure_connection(self) -> bool:
        return self.is_connected and self.mt5 is not None


class BrokerManager:
    """In-memory manager for saved Exness broker profiles."""

    def __init__(self):
        self.brokers: Dict[str, ExnessBroker] = {}
        self.active_broker: Optional[ExnessBroker] = None

    def add_broker(self, name: str, config: BrokerConfig) -> bool:
        """Register or replace a saved Exness broker."""
        try:
            if config.broker_type != BrokerType.EXNESS:
                logger.error(f"Unsupported broker type: {config.broker_type}")
                return False

            existing_broker = self.brokers.get(name)
            if existing_broker and existing_broker.is_connected:
                existing_broker.disconnect()
                if self.active_broker == existing_broker:
                    self.active_broker = None

            self.brokers[name] = ExnessBroker(config)
            logger.info(f"Broker added: {name}")
            return True
        except Exception as exc:
            logger.error(f"Failed to add broker: {exc}")
            return False

    def remove_broker(self, name: str) -> bool:
        """Remove a saved broker from the manager."""
        if name not in self.brokers:
            return False

        broker = self.brokers.pop(name)
        if broker.is_connected:
            broker.disconnect()
        if self.active_broker == broker:
            self.active_broker = None

        logger.info(f"Broker removed: {name}")
        return True

    def load_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> int:
        """Load persisted Exness profiles into the manager."""
        count = 0
        for name, profile_data in (profiles or {}).items():
            try:
                config = broker_config_from_dict(profile_data)
                if self.add_broker(name, config):
                    count += 1
            except Exception as exc:
                logger.error(f"Failed to load broker profile '{name}': {exc}")
        return count

    def connect_broker(self, name: str) -> bool:
        """Connect the named saved broker and mark it active."""
        try:
            if name not in self.brokers:
                logger.error(f"Broker does not exist: {name}")
                return False

            broker = self.brokers[name]
            if broker.connect():
                self.active_broker = broker
                logger.info(f"Active broker switched to: {name}")
                return True
            return False
        except Exception as exc:
            logger.error(f"Failed to connect broker: {exc}")
            return False

    def disconnect_all(self):
        """Disconnect all saved Exness sessions."""
        try:
            for broker in self.brokers.values():
                if broker.is_connected:
                    broker.disconnect()
            self.active_broker = None
            logger.info("All broker connections closed")
        except Exception as exc:
            logger.error(f"Failed to disconnect broker connections: {exc}")

    def get_broker_status(self) -> Dict:
        """Return connection status for all saved broker profiles."""
        status = {}
        for name, broker in self.brokers.items():
            status[name] = {
                "connected": broker.is_connected,
                "type": broker.config.broker_type.value,
                "sandbox": broker.config.sandbox,
                "last_heartbeat": broker.last_heartbeat.isoformat(),
            }

        status["active_broker"] = None
        if self.active_broker:
            for name, broker in self.brokers.items():
                if broker == self.active_broker:
                    status["active_broker"] = name
                    break
        return status

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str, price: Optional[float] = None) -> Dict:
        """Place an order through the active Exness connection."""
        if not self.active_broker:
            return {"success": False, "error": "No active Exness broker connection"}
        return self.active_broker.place_order(symbol, side, quantity, order_type, price)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order through the active Exness connection."""
        if not self.active_broker:
            return False
        return self.active_broker.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> Dict:
        """Fetch order status from the active Exness connection."""
        if not self.active_broker:
            return {}
        return self.active_broker.get_order_status(order_id)

    def get_positions(self) -> List[Dict]:
        """Fetch open positions from the active Exness connection."""
        if not self.active_broker:
            return []
        return self.active_broker.get_positions()

    def get_account_info(self) -> Dict:
        """Fetch account info from the active Exness connection."""
        if not self.active_broker:
            return {}
        return self.active_broker.get_account_info()

    def get_market_data(self, symbol: str) -> Dict:
        """Fetch market data from the active Exness connection."""
        if not self.active_broker:
            return {}
        return self.active_broker.get_market_data(symbol)


def create_broker_config(broker_type: str, **kwargs) -> BrokerConfig:
    """Create an Exness broker config from user input or persisted data."""
    broker_enum = BrokerType(broker_type.lower())
    if broker_enum != BrokerType.EXNESS:
        raise ValueError(f"Unsupported broker type for focused v1: {broker_type}")

    return BrokerConfig(
        broker_type=broker_enum,
        login=str(kwargs.get("login", "")),
        password=kwargs.get("password", ""),
        server=kwargs.get("server", ""),
        terminal_path=kwargs.get("terminal_path", ""),
        sandbox=kwargs.get("sandbox", False),
        account_id=kwargs.get("account_id", ""),
        timeout=kwargs.get("timeout", 30),
        max_retries=kwargs.get("max_retries", 3),
    )


def broker_config_to_dict(config: BrokerConfig) -> Dict[str, Any]:
    """Serialize a broker config for persistence."""
    data = asdict(config)
    data["broker_type"] = config.broker_type.value
    return data


def broker_config_from_dict(data: Dict[str, Any]) -> BrokerConfig:
    """Deserialize a persisted broker config."""
    payload = dict(data or {})
    broker_type = payload.pop("broker_type")
    return create_broker_config(broker_type=broker_type, **payload)
