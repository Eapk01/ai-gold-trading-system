from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.broker_interface import BrokerManager, BrokerType, ExnessBroker, create_broker_config


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

    def order_send(self, request):
        self.last_request = request
        return FakeResult(self.TRADE_RETCODE_DONE, order=123456, comment="done")

    def orders_get(self, ticket):
        return [SimpleNamespace(_asdict=lambda: {"ticket": ticket, "state": "open"})]

    def history_orders_get(self, ticket):
        return []

    def positions_get(self):
        return [SimpleNamespace(_asdict=lambda: {"symbol": "XAUUSD", "volume": 0.01})]

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


if __name__ == "__main__":
    unittest.main()
