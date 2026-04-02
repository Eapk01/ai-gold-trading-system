import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from src.config_utils import ConfigValidationError, get_default_config, load_config, save_config, validate_config


class ConfigUtilsTests(unittest.TestCase):
    def test_default_config_is_valid_without_exness_enabled(self):
        config = get_default_config()
        validate_config(config)

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
                "password": "secret",
                "server": "Exness-MT5Trial",
                "terminal_path": "C:\\Terminal64.exe",
                "sandbox": False,
                "api_key": "",
                "secret_key": "",
                "endpoint": "",
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
        config["brokers"]["exness"]["password"] = "secret"

        with self.assertRaises(ConfigValidationError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
