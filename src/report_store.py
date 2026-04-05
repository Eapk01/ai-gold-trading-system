"""Shared helpers for persisted report artifacts."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ReportDefinition:
    report_type: str
    file_prefix: str
    summary_key: str


class ReportStore:
    """List and load typed JSON reports using a shared shape."""

    def __init__(self, reports_directory: str = "reports") -> None:
        self.reports_directory = reports_directory

    def list_reports(self, definition: ReportDefinition, limit: int = 10) -> List[Dict[str, Any]]:
        pattern = str(Path(self.reports_directory) / f"{definition.file_prefix}_*.json")
        report_files = sorted(glob.glob(pattern), reverse=True)
        return [
            {
                "report_type": definition.report_type,
                "path": file_path,
                "name": Path(file_path).name,
                "timestamp": Path(file_path).stem.replace(f"{definition.file_prefix}_", ""),
            }
            for file_path in report_files[:limit]
        ]

    def load_report(self, definition: ReportDefinition, report_path: str) -> Dict[str, Any]:
        with open(report_path, "r", encoding="utf-8") as report_file:
            report = json.load(report_file)

        return {
            "report_type": definition.report_type,
            "path": report_path,
            "name": Path(report_path).name,
            "summary": report.get(definition.summary_key, {}),
            "report_payload": report,
        }
