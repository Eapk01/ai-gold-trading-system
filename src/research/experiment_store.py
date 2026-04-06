"""Persistence helpers for research experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentStore:
    """Save and load experiment artifacts from the configured research directory."""

    def __init__(self, experiments_directory: str = "reports/experiments") -> None:
        self.experiments_directory = Path(experiments_directory)
        self.experiments_directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExperimentStore":
        """Build a store from application config."""
        directory = config.get("research", {}).get("experiments_directory", "reports/experiments")
        return cls(directory)

    def save_result(self, result: Any, filename: str) -> Path:
        """Persist an experiment result as JSON."""
        target_path = self.resolve_path(filename)
        with target_path.open("w", encoding="utf-8") as file_obj:
            json.dump(result.to_dict(), file_obj, ensure_ascii=False, indent=2)
        return target_path

    def load_result(self, filename: str) -> Dict[str, Any]:
        """Load a previously saved experiment payload."""
        target_path = self.resolve_path(filename)
        with target_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def list_results(self, limit: int = 50, prefix: str | None = None) -> List[Path]:
        """Return saved experiment JSON paths newest-first."""
        pattern = f"{prefix}*.json" if prefix else "*.json"
        results = sorted(self.experiments_directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        return results[:limit]

    def resolve_path(self, filename_or_path: str | Path) -> Path:
        """Return an absolute experiment artifact path."""
        candidate = Path(filename_or_path)
        if candidate.is_absolute():
            return candidate
        if candidate.parent != Path("."):
            return candidate
        return self.experiments_directory / candidate
