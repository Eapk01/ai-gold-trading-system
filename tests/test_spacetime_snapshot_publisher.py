import unittest
from unittest.mock import patch

from src.spacetime_snapshot_publisher import SpacetimeSnapshotPublisher


class SpacetimeSnapshotPublisherTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "live_trading": {
                "spacetime_dashboard": {
                    "enabled": True,
                    "sender_script": r"C:\Users\Bruh\Desktop\Tests\scripts\send-snapshot.ts",
                    "sender_project_dir": r"C:\Users\Bruh\Desktop\Tests",
                    "npx_command": r"C:\Program Files\nodejs\npx.cmd",
                    "uri": "https://maincloud.spacetimedb.com",
                    "database_name": "trading-bot-demo-bruh-20260419",
                }
            }
        }
        self.publisher = SpacetimeSnapshotPublisher(self.config)

    def test_build_payload_for_active_buy_position(self):
        payload = self.publisher.build_payload(
            symbol="XAUUSDm",
            timeframe="5m",
            latest_signal={"price": 2350.42, "confidence": 0.74},
            positions=[
                {
                    "ticket": "42",
                    "type": 0,
                    "price_open": 2348.9,
                    "price_current": 2350.42,
                    "sl": 2344.2,
                    "tp": 2358.7,
                    "profit": 18.25,
                }
            ],
            account_info={"equity": 10510.35},
            event_message="Opened buy position",
        )

        self.assertEqual(payload["position_id"], "42")
        self.assertEqual(payload["symbol"], "XAUUSD")
        self.assertEqual(payload["timeframe"], "M5")
        self.assertEqual(payload["side"], "BUY")
        self.assertEqual(payload["status"], "active")
        self.assertEqual(payload["entry_price"], 2348.9)
        self.assertEqual(payload["pnl"], 18.25)
        self.assertEqual(payload["equity"], 10510.35)

    def test_build_payload_for_active_sell_position(self):
        payload = self.publisher.build_payload(
            symbol="XAUUSD",
            timeframe="1h",
            latest_signal={"price": 2330.0, "confidence": 0.66},
            positions=[
                {
                    "ticket": "99",
                    "type": 1,
                    "price_open": 2335.0,
                    "price_current": 2330.0,
                    "sl": 2340.0,
                    "tp": 2320.0,
                    "profit": 22.5,
                }
            ],
            account_info={"equity": 10022.5},
            event_message="Holding existing sell position",
        )

        self.assertEqual(payload["side"], "SELL")
        self.assertEqual(payload["status"], "active")
        self.assertEqual(payload["timeframe"], "H1")
        self.assertEqual(payload["take_profit"], 2320.0)

    def test_build_payload_for_idle_state(self):
        payload = self.publisher.build_payload(
            symbol="XAUUSDm",
            timeframe="5m",
            latest_signal={"price": 2350.42, "confidence": 0.74},
            positions=[],
            account_info={},
            event_message="Auto trader stopped",
        )

        self.assertEqual(payload["position_id"], 1)
        self.assertEqual(payload["side"], "NONE")
        self.assertEqual(payload["status"], "idle")
        self.assertEqual(payload["price"], 2350.42)
        self.assertEqual(payload["equity"], 0.0)

    @patch("src.spacetime_snapshot_publisher.subprocess.run")
    def test_publish_payload_invokes_external_sender(self, mock_run):
        payload = {"symbol": "XAUUSD", "timeframe": "M5"}

        success = self.publisher.publish_payload(payload)

        self.assertTrue(success)
        self.assertEqual(mock_run.call_count, 1)
        args, kwargs = mock_run.call_args
        self.assertEqual(
            args[0],
            [
                r"C:\Program Files\nodejs\npx.cmd",
                "tsx",
                r"C:\Users\Bruh\Desktop\Tests\scripts\send-snapshot.ts",
            ],
        )
        self.assertEqual(kwargs["cwd"], r"C:\Users\Bruh\Desktop\Tests")
        self.assertEqual(kwargs["env"]["VITE_SPACETIME_URI"], "https://maincloud.spacetimedb.com")
        self.assertEqual(kwargs["env"]["VITE_SPACETIME_DB_NAME"], "trading-bot-demo-bruh-20260419")
        self.assertEqual(kwargs["env"]["BOT_SNAPSHOT"], '{"symbol": "XAUUSD", "timeframe": "M5"}')
        self.assertTrue(kwargs["check"])


if __name__ == "__main__":
    unittest.main()
