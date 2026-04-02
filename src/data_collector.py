"""
Local dataset collector for the focused v1 workflow.
Imports MetaTrader-style OHLCV exports from the project dataset directory.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover - exercised only in minimal environments
    create_engine = None


class DataCollector:
    """Load, normalize, validate, and persist local market datasets."""

    MT_REQUIRED_COLUMNS = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"]
    MT_OPTIONAL_VOLUME_COLUMNS = ["<TICKVOL>", "<VOL>"]

    def __init__(self, config: Dict):
        self.config = config
        self.symbol = config["trading"]["symbol"]
        self.timeframe = config["trading"]["timeframe"]
        self.dataset_directory = Path(config["data_sources"].get("dataset_directory", "data/imports"))
        self.min_rows = int(config["data_sources"].get("min_rows", 100))
        self.database_path = config["database"]["path"]
        self.db_engine = create_engine(f"sqlite:///{self.database_path}") if create_engine else None

        logger.info(f"Data collector initialized - symbol: {self.symbol}")

    def find_first_dataset_file(self) -> Path:
        """Return the first CSV file in the configured dataset directory."""
        self.dataset_directory.mkdir(parents=True, exist_ok=True)
        candidates = sorted(self.dataset_directory.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files were found in dataset directory: {self.dataset_directory}"
            )
        selected = candidates[0]
        logger.info(f"Selected dataset file: {selected}")
        return selected

    def import_default_dataset(self) -> Tuple[Path, pd.DataFrame]:
        """Load the first CSV dataset from the configured directory."""
        file_path = self.find_first_dataset_file()
        return file_path, self.import_mt_csv(file_path)

    def import_mt_csv(self, file_path: Path | str) -> pd.DataFrame:
        """Import a MetaTrader-style OHLCV export and normalize it."""
        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")

        delimiter = self._detect_delimiter(path)
        raw_data = pd.read_csv(path, encoding="utf-8", sep=delimiter)
        normalized = self._normalize_mt_export(raw_data)

        if len(normalized) < self.min_rows:
            raise ValueError(
                f"Imported dataset is too small: {len(normalized)} rows found, at least {self.min_rows} required"
            )

        logger.info(f"Imported local CSV dataset: {path} ({len(normalized)} rows)")
        return normalized

    def _normalize_mt_export(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize MetaTrader export columns into the internal OHLCV schema."""
        missing_columns = [column for column in self.MT_REQUIRED_COLUMNS if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        timestamp_series = (
            data["<DATE>"].astype(str).str.strip() + " " + data["<TIME>"].astype(str).str.strip()
        )

        normalized = pd.DataFrame()
        normalized["Timestamp"] = pd.to_datetime(timestamp_series, errors="coerce")
        normalized["Open"] = pd.to_numeric(data["<OPEN>"], errors="coerce")
        normalized["High"] = pd.to_numeric(data["<HIGH>"], errors="coerce")
        normalized["Low"] = pd.to_numeric(data["<LOW>"], errors="coerce")
        normalized["Close"] = pd.to_numeric(data["<CLOSE>"], errors="coerce")

        volume_column = next(
            (column for column in self.MT_OPTIONAL_VOLUME_COLUMNS if column in data.columns),
            None,
        )
        if volume_column:
            normalized["Volume"] = pd.to_numeric(data[volume_column], errors="coerce").fillna(0.0)
        else:
            normalized["Volume"] = 0.0

        invalid_timestamp_count = int(normalized["Timestamp"].isna().sum())
        if invalid_timestamp_count:
            logger.warning(f"Dropping rows with invalid timestamps: {invalid_timestamp_count}")

        normalized = normalized.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
        normalized = normalized.sort_values("Timestamp")
        normalized = normalized.drop_duplicates(subset=["Timestamp"], keep="last")
        normalized = normalized.set_index("Timestamp")
        normalized.index.name = "DateTime"
        normalized["Timestamp"] = normalized.index
        return normalized

    def save_data_to_db(self, data: pd.DataFrame, table_name: str):
        """Persist a dataframe to SQLite."""
        try:
            if self.db_engine is not None:
                data.to_sql(table_name, self.db_engine, if_exists="replace", index=True)
            else:
                with sqlite3.connect(self.database_path) as connection:
                    data.to_sql(table_name, connection, if_exists="replace", index=True)
            logger.info(f"Data saved to database table: {table_name}")
        except Exception as exc:
            logger.error(f"Failed to save data to database: {exc}")

    def load_data_from_db(
        self,
        table_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data from SQLite."""
        try:
            query = f"SELECT * FROM {table_name}"
            if start_date and end_date:
                query += f" WHERE DateTime BETWEEN '{start_date}' AND '{end_date}'"

            if self.db_engine is not None:
                data = pd.read_sql(query, self.db_engine, index_col="DateTime")
            else:
                with sqlite3.connect(self.database_path) as connection:
                    data = pd.read_sql(query, connection, index_col="DateTime")

            logger.info(f"Loaded data from database: {len(data)} rows")
            return data
        except Exception as exc:
            logger.error(f"Failed to load data from database: {exc}")
            return pd.DataFrame()

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate imported OHLCV data quality."""
        issues: List[str] = []

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        null_counts = data.isnull().sum()
        if null_counts.any():
            issues.append(f"Null values detected: {null_counts[null_counts > 0].to_dict()}")

        if "High" in data.columns and "Low" in data.columns:
            invalid_price_logic = int((data["High"] < data["Low"]).sum())
            if invalid_price_logic > 0:
                issues.append(f"Invalid high/low relationships: {invalid_price_logic} rows")

        duplicate_count = int(data.index.duplicated().sum())
        if duplicate_count > 0:
            issues.append(f"Duplicate timestamps: {duplicate_count}")

        is_valid = len(issues) == 0
        if is_valid:
            logger.info("Data quality validation passed")
        else:
            logger.warning(f"Data quality issues detected: {issues}")

        return is_valid, issues

    def _detect_delimiter(self, path: Path) -> str:
        """Detect the CSV delimiter used by the dataset file."""
        with path.open("r", encoding="utf-8") as file_obj:
            sample = file_obj.read(1024)
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
