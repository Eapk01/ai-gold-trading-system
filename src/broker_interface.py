"""
券商接口模块
与实际券商API集成的订单管理系统
支持多种券商平台
"""

import requests
import websocket
import json
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from abc import ABC, abstractmethod
import hmac
import hashlib
import base64
from importlib import import_module


class BrokerType(Enum):
    """支持的券商类型"""
    MT5 = "mt5"
    EXNESS = "exness"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "ib"
    OANDA = "oanda"
    BINANCE = "binance"
    BYBIT = "bybit"
    PAPER = "paper"  # 模拟交易


@dataclass
class BrokerConfig:
    """券商配置"""
    broker_type: BrokerType
    api_key: str = ""
    secret_key: str = ""
    login: str = ""
    password: str = ""
    server: str = ""
    terminal_path: str = ""
    endpoint: str = ""
    sandbox: bool = True
    account_id: str = ""
    timeout: int = 30
    max_retries: int = 3


class BrokerInterface(ABC):
    """券商接口抽象基类"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.is_connected = False
        self.last_heartbeat = datetime.now()
        self.callbacks: Dict[str, Callable] = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """连接到券商"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str, price: Optional[float] = None) -> Dict:
        """下单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """获取订单状态"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """获取持仓"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        """获取市场数据"""
        pass
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
        self.callbacks[event_type] = callback


def _import_metatrader5():
    """Import MetaTrader5 lazily so the dependency remains optional."""
    try:
        return import_module("MetaTrader5")
    except ImportError:
        return None


class AlpacaBroker(BrokerInterface):
    """Alpaca券商接口"""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.base_url = "https://paper-api.alpaca.markets" if config.sandbox else "https://api.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.secret_key,
            "Content-Type": "application/json"
        }
        self.ws = None
        self.ws_thread = None
        
    def connect(self) -> bool:
        """连接到Alpaca"""
        try:
            # 测试API连接
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=self.config.timeout)
            
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Alpaca API连接成功")
                
                # 启动WebSocket连接
                self._start_websocket()
                return True
            else:
                logger.error(f"Alpaca API连接失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"连接Alpaca失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        try:
            self.is_connected = False
            
            if self.ws:
                self.ws.close()
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
                
            logger.info("Alpaca连接已断开")
            
        except Exception as e:
            logger.error(f"断开Alpaca连接失败: {e}")
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str, price: Optional[float] = None) -> Dict:
        """在Alpaca下单"""
        try:
            order_data = {
                "symbol": symbol,
                "qty": quantity,
                "side": side,
                "type": order_type,
                "time_in_force": "GTC"
            }
            
            if price and order_type in ["limit", "stop_limit"]:
                order_data["limit_price"] = price
                
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 201:
                order_result = response.json()
                logger.info(f"Alpaca订单已提交: {order_result['id']}")
                return {
                    "success": True,
                    "order_id": order_result["id"],
                    "status": order_result["status"],
                    "data": order_result
                }
            else:
                error_msg = response.json().get("message", "未知错误")
                logger.error(f"Alpaca订单失败: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Alpaca下单异常: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销Alpaca订单"""
        try:
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 204:
                logger.info(f"Alpaca订单已撤销: {order_id}")
                return True
            else:
                logger.error(f"Alpaca撤单失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Alpaca撤单异常: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """获取Alpaca订单状态"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取Alpaca订单状态异常: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """获取Alpaca持仓"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取Alpaca持仓异常: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """获取Alpaca账户信息"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取Alpaca账户信息异常: {e}")
            return {}
    
    def get_market_data(self, symbol: str) -> Dict:
        """获取Alpaca市场数据"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/stocks/{symbol}/quotes/latest",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取Alpaca市场数据异常: {e}")
            return {}
    
    def _start_websocket(self):
        """启动WebSocket连接"""
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._handle_websocket_message(data)
                except Exception as e:
                    logger.error(f"处理WebSocket消息失败: {e}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket错误: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.warning("WebSocket连接已关闭")
            
            def on_open(ws):
                logger.info("WebSocket连接已建立")
                # 订阅交易更新
                auth_msg = {
                    "action": "auth",
                    "key": self.config.api_key,
                    "secret": self.config.secret_key
                }
                ws.send(json.dumps(auth_msg))
            
            ws_url = "wss://paper-api.alpaca.markets/stream" if self.config.sandbox else "wss://api.alpaca.markets/stream"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            logger.error(f"启动WebSocket失败: {e}")
    
    def _handle_websocket_message(self, data: Dict):
        """处理WebSocket消息"""
        try:
            msg_type = data.get("stream")
            
            if msg_type == "trade_updates":
                # 订单更新
                order_data = data.get("data", {})
                if "order_update" in self.callbacks:
                    self.callbacks["order_update"](order_data)
                    
            elif msg_type == "quotes":
                # 报价更新
                quote_data = data.get("data", {})
                if "market_data" in self.callbacks:
                    self.callbacks["market_data"](quote_data)
                    
        except Exception as e:
            logger.error(f"处理WebSocket消息异常: {e}")


class OANDABroker(BrokerInterface):
    """OANDA券商接口"""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.base_url = "https://api-fxpractice.oanda.com" if config.sandbox else "https://api-fxtrade.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
    def connect(self) -> bool:
        """连接到OANDA"""
        try:
            response = requests.get(
                f"{self.base_url}/v3/accounts/{self.config.account_id}",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                self.is_connected = True
                logger.info("OANDA API连接成功")
                return True
            else:
                logger.error(f"OANDA API连接失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"连接OANDA失败: {e}")
            return False
    
    def disconnect(self):
        """断开OANDA连接"""
        self.is_connected = False
        logger.info("OANDA连接已断开")
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str, price: Optional[float] = None) -> Dict:
        """在OANDA下单"""
        try:
            units = int(quantity) if side == "buy" else -int(quantity)
            
            order_data = {
                "order": {
                    "instrument": symbol,
                    "units": str(units),
                    "type": order_type.upper(),
                    "timeInForce": "GTC"
                }
            }
            
            if price and order_type.lower() in ["limit", "stop"]:
                order_data["order"]["price"] = str(price)
            
            response = requests.post(
                f"{self.base_url}/v3/accounts/{self.config.account_id}/orders",
                headers=self.headers,
                json=order_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 201:
                result = response.json()
                order_result = result.get("orderCreateTransaction", {})
                logger.info(f"OANDA订单已提交: {order_result.get('id')}")
                return {
                    "success": True,
                    "order_id": order_result.get("id"),
                    "status": "pending",
                    "data": order_result
                }
            else:
                error_msg = response.json().get("errorMessage", "未知错误")
                logger.error(f"OANDA订单失败: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"OANDA下单异常: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销OANDA订单"""
        try:
            response = requests.put(
                f"{self.base_url}/v3/accounts/{self.config.account_id}/orders/{order_id}/cancel",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"OANDA订单已撤销: {order_id}")
                return True
            else:
                logger.error(f"OANDA撤单失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"OANDA撤单异常: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """获取OANDA订单状态"""
        try:
            response = requests.get(
                f"{self.base_url}/v3/accounts/{self.config.account_id}/orders/{order_id}",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("order", {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取OANDA订单状态异常: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """获取OANDA持仓"""
        try:
            response = requests.get(
                f"{self.base_url}/v3/accounts/{self.config.account_id}/positions",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                positions = response.json().get("positions", [])
                return [pos for pos in positions if float(pos.get("long", {}).get("units", 0)) != 0 or float(pos.get("short", {}).get("units", 0)) != 0]
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取OANDA持仓异常: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """获取OANDA账户信息"""
        try:
            response = requests.get(
                f"{self.base_url}/v3/accounts/{self.config.account_id}",
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("account", {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取OANDA账户信息异常: {e}")
            return {}
    
    def get_market_data(self, symbol: str) -> Dict:
        """获取OANDA市场数据"""
        try:
            response = requests.get(
                f"{self.base_url}/v3/instruments/{symbol}/candles",
                headers=self.headers,
                params={"count": 1, "granularity": "M1"},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                candles = response.json().get("candles", [])
                if candles:
                    latest = candles[-1]
                    return {
                        "symbol": symbol,
                        "bid": float(latest["bid"]["c"]),
                        "ask": float(latest["ask"]["c"]),
                        "timestamp": latest["time"]
                    }
            return {}
                
        except Exception as e:
            logger.error(f"获取OANDA市场数据异常: {e}")
            return {}


class ExnessBroker(BrokerInterface):
    """Exness broker adapter using the MetaTrader5 terminal as a v1 integration path."""

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
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

        except Exception as e:
            logger.error(f"Failed to connect Exness broker: {e}")
            return False

    def disconnect(self):
        """Disconnect the MetaTrader5 terminal session."""
        try:
            if self.mt5:
                self.mt5.shutdown()
            self.is_connected = False
            logger.info("Exness connection closed")
        except Exception as e:
            logger.error(f"Failed to disconnect Exness broker: {e}")

    def place_order(self, symbol: str, side: str, quantity: float,
                   order_type: str, price: Optional[float] = None) -> Dict:
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

            success = result.retcode in {
                getattr(self.mt5, "TRADE_RETCODE_DONE", -1),
                getattr(self.mt5, "TRADE_RETCODE_PLACED", -2),
            }
            payload = result._asdict() if hasattr(result, "_asdict") else {"retcode": getattr(result, "retcode", None)}

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

        except Exception as e:
            logger.error(f"Exness order placement failed: {e}")
            return {"success": False, "error": str(e)}

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
        except Exception as e:
            logger.error(f"Exness cancel order failed: {e}")
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
        except Exception as e:
            logger.error(f"Exness get order status failed: {e}")
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
        except Exception as e:
            logger.error(f"Exness get positions failed: {e}")
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
        except Exception as e:
            logger.error(f"Exness get account info failed: {e}")
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
        except Exception as e:
            logger.error(f"Exness get market data failed: {e}")
            return {}

    def _ensure_connection(self) -> bool:
        return self.is_connected and self.mt5 is not None


class BrokerManager:
    """券商管理器"""
    
    def __init__(self):
        self.brokers: Dict[str, BrokerInterface] = {}
        self.active_broker: Optional[BrokerInterface] = None
        self.config_map: Dict[BrokerType, type] = {
            BrokerType.EXNESS: ExnessBroker,
            BrokerType.ALPACA: AlpacaBroker,
            BrokerType.OANDA: OANDABroker,
            # 可以添加更多券商
        }
        
    def add_broker(self, name: str, config: BrokerConfig) -> bool:
        """添加券商"""
        try:
            if config.broker_type not in self.config_map:
                logger.error(f"Unsupported broker type: {config.broker_type}")
                return False
            
            broker_class = self.config_map[config.broker_type]
            broker = broker_class(config)
            
            self.brokers[name] = broker
            logger.info(f"Broker added: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add broker: {e}")
            return False
    
    def connect_broker(self, name: str) -> bool:
        """连接指定券商"""
        try:
            if name not in self.brokers:
                logger.error(f"Broker does not exist: {name}")
                return False
            
            broker = self.brokers[name]
            if broker.connect():
                self.active_broker = broker
                logger.info(f"已切换到券商: {name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"连接券商失败: {e}")
            return False
    
    def disconnect_all(self):
        """断开所有券商连接"""
        try:
            for broker in self.brokers.values():
                if broker.is_connected:
                    broker.disconnect()
            
            self.active_broker = None
            logger.info("所有券商连接已断开")
            
        except Exception as e:
            logger.error(f"断开券商连接失败: {e}")
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str, price: Optional[float] = None) -> Dict:
        """通过活跃券商下单"""
        if not self.active_broker:
            return {"success": False, "error": "没有活跃的券商连接"}
        
        return self.active_broker.place_order(symbol, side, quantity, order_type, price)
    
    def cancel_order(self, order_id: str) -> bool:
        """通过活跃券商撤单"""
        if not self.active_broker:
            return False
        
        return self.active_broker.cancel_order(order_id)
    
    def get_order_status(self, order_id: str) -> Dict:
        """获取订单状态"""
        if not self.active_broker:
            return {}
        
        return self.active_broker.get_order_status(order_id)
    
    def get_positions(self) -> List[Dict]:
        """获取持仓"""
        if not self.active_broker:
            return []
        
        return self.active_broker.get_positions()
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        if not self.active_broker:
            return {}
        
        return self.active_broker.get_account_info()
    
    def get_market_data(self, symbol: str) -> Dict:
        """获取市场数据"""
        if not self.active_broker:
            return {}
        
        return self.active_broker.get_market_data(symbol)
    
    def get_broker_status(self) -> Dict:
        """获取券商状态"""
        status = {}
        
        for name, broker in self.brokers.items():
            status[name] = {
                "connected": broker.is_connected,
                "type": broker.config.broker_type.value,
                "sandbox": broker.config.sandbox,
                "last_heartbeat": broker.last_heartbeat.isoformat()
            }
        
        status["active_broker"] = None
        if self.active_broker:
            for name, broker in self.brokers.items():
                if broker == self.active_broker:
                    status["active_broker"] = name
                    break
        
        return status
    
    def register_callback(self, event_type: str, callback: Callable):
        """为所有券商注册回调"""
        for broker in self.brokers.values():
            broker.register_callback(event_type, callback)


# 工厂函数
def create_broker_config(broker_type: str, **kwargs) -> BrokerConfig:
    """创建券商配置"""
    broker_enum = BrokerType(broker_type.lower())
    
    return BrokerConfig(
        broker_type=broker_enum,
        api_key=kwargs.get('api_key', ''),
        secret_key=kwargs.get('secret_key', ''),
        login=str(kwargs.get('login', '')),
        password=kwargs.get('password', ''),
        server=kwargs.get('server', ''),
        terminal_path=kwargs.get('terminal_path', ''),
        endpoint=kwargs.get('endpoint', ''),
        sandbox=kwargs.get('sandbox', True),
        account_id=kwargs.get('account_id', ''),
        timeout=kwargs.get('timeout', 30),
        max_retries=kwargs.get('max_retries', 3)
    ) 
