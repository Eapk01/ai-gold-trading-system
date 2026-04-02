from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.config_utils import get_default_config
from src.data_collector import DataCollector


class DataCollectorImportTests(unittest.TestCase):
    def _build_collector(self, temp_dir: str) -> DataCollector:
        config = get_default_config()
        config["database"]["path"] = str(Path(temp_dir) / "trading_system.db")
        config["data_sources"]["dataset_directory"] = str(Path(temp_dir) / "imports")
        config["data_sources"]["min_rows"] = 2
        return DataCollector(config)

    def _write_mt_dataset(self, path: Path, rows: str):
        path.write_text(
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n" + rows,
            encoding="utf-8",
        )

    def test_import_mt_csv_normalizes_expected_columns(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "gold.csv"
            self._write_mt_dataset(
                csv_path,
                "2025.01.01\t23:05:00\t2625.179\t2625.839\t2624.575\t2625.230\t344\t0\t160\n"
                "2025.01.01\t23:10:00\t2625.249\t2625.418\t2624.374\t2624.789\t318\t0\t160\n",
            )

            collector = self._build_collector(temp_dir)
            data = collector.import_mt_csv(csv_path)

            self.assertEqual(list(data.columns), ["Open", "High", "Low", "Close", "Volume", "Timestamp"])
            self.assertEqual(len(data), 2)
            self.assertEqual(str(data.index[0]), "2025-01-01 23:05:00")
            self.assertEqual(float(data["Volume"].iloc[0]), 344.0)

    def test_import_mt_csv_uses_vol_if_tickvol_missing(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "gold.csv"
            csv_path.write_text(
                "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<VOL>\t<SPREAD>\n"
                "2025.01.01\t23:05:00\t2625.179\t2625.839\t2624.575\t2625.230\t9\t160\n"
                "2025.01.01\t23:10:00\t2625.249\t2625.418\t2624.374\t2624.789\t10\t160\n",
                encoding="utf-8",
            )

            collector = self._build_collector(temp_dir)
            data = collector.import_mt_csv(csv_path)

            self.assertEqual(float(data["Volume"].iloc[0]), 9.0)

    def test_import_mt_csv_rejects_missing_required_columns(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "gold.csv"
            csv_path.write_text(
                "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<CLOSE>\n"
                "2025.01.01\t23:05:00\t2625.179\t2625.839\t2625.230\n"
                "2025.01.01\t23:10:00\t2625.249\t2625.418\t2624.789\n",
                encoding="utf-8",
            )

            collector = self._build_collector(temp_dir)
            with self.assertRaises(ValueError):
                collector.import_mt_csv(csv_path)

    def test_import_default_dataset_uses_first_csv_in_directory(self):
        with TemporaryDirectory() as temp_dir:
            imports_dir = Path(temp_dir) / "imports"
            imports_dir.mkdir(parents=True, exist_ok=True)
            second_file = imports_dir / "b_second.csv"
            first_file = imports_dir / "a_first.csv"

            self._write_mt_dataset(
                second_file,
                "2025.01.01\t23:10:00\t2625.249\t2625.418\t2624.374\t2624.789\t318\t0\t160\n"
                "2025.01.01\t23:15:00\t2624.700\t2625.000\t2624.200\t2624.500\t319\t0\t160\n",
            )
            self._write_mt_dataset(
                first_file,
                "2025.01.01\t23:05:00\t2625.179\t2625.839\t2624.575\t2625.230\t344\t0\t160\n"
                "2025.01.01\t23:10:00\t2625.249\t2625.418\t2624.374\t2624.789\t318\t0\t160\n",
            )

            collector = self._build_collector(temp_dir)
            selected_path, data = collector.import_default_dataset()

            self.assertEqual(selected_path.name, "a_first.csv")
            self.assertEqual(str(data.index[0]), "2025-01-01 23:05:00")


if __name__ == "__main__":
    unittest.main()
