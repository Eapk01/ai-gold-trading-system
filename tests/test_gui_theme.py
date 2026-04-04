import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gui.theme import DEFAULT_THEME_MODE, load_theme_mode, save_theme_mode


class ThemePreferenceTests(unittest.TestCase):
    def test_missing_preferences_file_falls_back_to_default(self):
        with TemporaryDirectory() as temp_dir:
            preferences_path = Path(temp_dir) / "ui_preferences.json"
            self.assertEqual(load_theme_mode(preferences_path), DEFAULT_THEME_MODE)

    def test_saved_light_preference_restores_correctly(self):
        with TemporaryDirectory() as temp_dir:
            preferences_path = Path(temp_dir) / "ui_preferences.json"
            save_theme_mode("light", preferences_path)
            self.assertEqual(load_theme_mode(preferences_path), "light")

    def test_invalid_preference_value_falls_back_safely(self):
        with TemporaryDirectory() as temp_dir:
            preferences_path = Path(temp_dir) / "ui_preferences.json"
            preferences_path.write_text(json.dumps({"theme_mode": "sepia"}), encoding="utf-8")
            self.assertEqual(load_theme_mode(preferences_path), DEFAULT_THEME_MODE)


if __name__ == "__main__":
    unittest.main()
