from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.broker_interface import (
    BrokerManager,
    BrokerType,
    ExnessBroker,
    broker_config_from_dict,
    broker_config_to_dict,
    create_broker_config,
)


class FakeResult:
    def __init__(self, retcode, **extra):
        self.retcode = retcode
        self.extra = extra

    def _asdict(self):
        data = {"retcode": self.retcode}
        data.update(self.extra)
        return data


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 6
    TRADE_ACTION_SLTP = 7
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008

    def initialize(self, **kwargs):
        self.initialized_with = kwargs
        return True

    def login(self, login, password, server):
        self.login_args = (login, password, server)
        return True

    def shutdown(self):
        return True

    def last_error(self):
        return (0, "ok")

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1999.5, ask=2000.0, last=1999.8, time=1710000000)

    def symbol_info(self, symbol):
        return SimpleNamespace(volume_min=0.01, volume_max=100.0, volume_step=0.01)

    def order_send(self, request):
        self.last_request = request
        return FakeResult(self.TRADE_RETCODE_DONE, order=123456, comment="done")

    def orders_get(self, ticket):
        return [SimpleNamespace(_asdict=lambda: {"ticket": ticket, "state": "open"})]

    def history_orders_get(self, ticket):
        return []

    def positions_get(self, ticket=None):
        position = SimpleNamespace(
            symbol="XAUUSD",
            volume=0.01,
            type=self.ORDER_TYPE_BUY,
            sl=1990.0,
            tp=2010.0,
        )
        return [position]

    def account_info(self):
        return SimpleNamespace(_asdict=lambda: {"login": 123456, "balance": 10000.0})


class BrokerInterfaceTests(unittest.TestCase):
    def test_create_broker_config_supports_exness_fields(self):
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
            terminal_path="C:\\Terminal64.exe",
        )

        self.assertEqual(config.broker_type, BrokerType.EXNESS)
        self.assertEqual(config.login, "123456")
        self.assertEqual(config.server, "Exness-MT5Real")
        self.assertEqual(config.terminal_path, "C:\\Terminal64.exe")

    @patch("src.broker_interface._import_metatrader5", return_value=FakeMT5())
    def test_exness_broker_connects_and_places_market_order(self, _mock_mt5):
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
        )
        broker = ExnessBroker(config)

        self.assertTrue(broker.connect())
        result = broker.place_order("XAUUSD", "buy", 0.01, "market")

        self.assertTrue(result["success"])
        self.assertEqual(result["order_id"], "123456")
        self.assertEqual(result["status"], "filled")

    @patch("src.broker_interface._import_metatrader5", return_value=FakeMT5())
    def test_exness_broker_updates_position_protection(self, _mock_mt5):
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
        )
        broker = ExnessBroker(config)

        self.assertTrue(broker.connect())
        result = broker.update_position_protection("77", stop_loss=1995.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["position_ticket"], "77")
        self.assertEqual(result["stop_loss"], 1995.0)
        self.assertEqual(result["take_profit"], 2010.0)

    @patch("src.broker_interface._import_metatrader5", return_value=FakeMT5())
    def test_exness_broker_normalizes_volume_to_symbol_step(self, _mock_mt5):
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
        )
        broker = ExnessBroker(config)

        self.assertTrue(broker.connect())
        broker.mt5.symbol_info = lambda symbol: SimpleNamespace(volume_min=0.1, volume_max=100.0, volume_step=0.1)
        result = broker.place_order("XAUUSD", "buy", 0.15, "market")

        self.assertTrue(result["success"])
        self.assertEqual(broker.mt5.last_request["volume"], 0.1)

    @patch("src.broker_interface._import_metatrader5", return_value=FakeMT5())
    def test_exness_broker_rejects_volume_below_symbol_minimum(self, _mock_mt5):
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
        )
        broker = ExnessBroker(config)

        self.assertTrue(broker.connect())
        broker.mt5.symbol_info = lambda symbol: SimpleNamespace(volume_min=0.1, volume_max=100.0, volume_step=0.1)
        result = broker.place_order("XAUUSD", "buy", 0.01, "market")

        self.assertFalse(result["success"])
        self.assertIn("below the minimum", result["error"])
        self.assertFalse(hasattr(broker.mt5, "last_request"))

    def test_broker_manager_registers_exness(self):
        manager = BrokerManager()
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
        )

        self.assertTrue(manager.add_broker("exness-main", config))
        status = manager.get_broker_status()

        self.assertIn("exness-main", status)
        self.assertEqual(status["exness-main"]["type"], "exness")

    def test_broker_config_round_trip_serialization(self):
        config = create_broker_config(
            broker_type="exness",
            login="123456",
            password="secret",
            server="Exness-MT5Real",
            terminal_path="C:\\Terminal64.exe",
            sandbox=False,
        )

        payload = broker_config_to_dict(config)
        restored = broker_config_from_dict(payload)

        self.assertEqual(restored.broker_type, BrokerType.EXNESS)
        self.assertEqual(restored.login, "123456")
        self.assertEqual(restored.terminal_path, "C:\\Terminal64.exe")

    def test_broker_manager_loads_profiles(self):
        manager = BrokerManager()
        profiles = {
            "saved-exness": {
                "broker_type": "exness",
                "login": "123456",
                "server": "Exness-MT5Real",
                "terminal_path": "",
                "sandbox": False,
                "account_id": "",
                "timeout": 30,
                "max_retries": 3,
            }
        }

        self.assertEqual(manager.load_profiles(profiles), 1)
        self.assertIn("saved-exness", manager.get_broker_status())


if __name__ == "__main__":
    unittest.main()
