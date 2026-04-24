import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from src.config_utils import (
    ConfigValidationError,
    get_default_config,
    get_effective_confidence_threshold,
    get_target_column,
    load_config,
    save_config,
    validate_config,
)


class ConfigUtilsTests(unittest.TestCase):
    def test_default_config_is_valid_without_exness_enabled(self):
        config = get_default_config()
        validate_config(config)
        self.assertEqual(get_target_column(config), "Future_Direction_1")
        self.assertIn("spacetime_dashboard", config["live_trading"])

    def test_validate_config_rejects_invalid_position_size(self):
        config = get_default_config()
        config["trading"]["position_size"] = -1

        with self.assertRaises(ConfigValidationError):
            validate_config(config)

    def test_save_and_load_config_preserves_broker_profiles(self):
        config = get_default_config()
        config["brokers"]["profiles"] = {
            "demo-exness": {
                "broker_type": "exness",
                "login": "123456",
                "server": "Exness-MT5Trial",
                "terminal_path": "C:\\Terminal64.exe",
                "sandbox": False,
                "account_id": "",
                "timeout": 30,
                "max_retries": 3,
            }
        }
        config["brokers"]["default_profile"] = "demo-exness"

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            save_config(config, str(config_path))
            loaded = load_config(str(config_path))

        self.assertIn("demo-exness", loaded["brokers"]["profiles"])
        self.assertEqual(loaded["brokers"]["default_profile"], "demo-exness")

    @patch("src.config_utils.find_spec", return_value=None)
    def test_validate_config_requires_mt5_when_exness_enabled(self, _mock_find_spec):
        config = get_default_config()
        config["brokers"]["exness"]["enabled"] = True
        config["brokers"]["exness"]["server"] = "Exness-MT5Real"
        config["brokers"]["exness"]["login"] = "123456"

        with self.assertRaises(ConfigValidationError):
            validate_config(config)

    def test_effective_confidence_threshold_prefers_mode_override(self):
        config = get_default_config()
        config["trading"]["confidence_threshold"] = 0.61
        config["backtest"]["signal_confidence_threshold"] = 0.72
        config["live_trading"]["signal_confidence_threshold"] = 0.67

        self.assertEqual(get_effective_confidence_threshold(config, "backtest"), 0.72)
        self.assertEqual(get_effective_confidence_threshold(config, "live_trading"), 0.67)

    def test_validate_config_rejects_invalid_stage5_defaults(self):
        config = get_default_config()
        config["research"]["stage5_defaults"] = {
            "target_ids": [],
            "feature_sets": [],
            "presets": ["conservative"],
        }

        with self.assertRaises(ConfigValidationError):
            validate_config(config)

    def test_validate_config_rejects_invalid_research_defaults(self):
        config = get_default_config()
        config["research"]["defaults"]["common"]["threshold_list"] = []

        with self.assertRaises(ConfigValidationError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
