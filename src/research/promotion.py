"""Promotion boundary between research candidates and production artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class PromotionManifest:
    """Metadata recorded when a candidate artifact is promoted to production."""

    experiment_name: str
    source_model_path: str
    promoted_model_path: str
    target_column: str
    feature_set_name: str = ""
    selected_threshold: float | None = None
    integrity: Dict[str, Any] = field(default_factory=dict)
    integrity_artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def get_default_promoted_model_path(models_directory: str, experiment_name: str) -> Path:
    """Return the default destination path for a promoted model artifact."""
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in experiment_name).strip("_")
    return Path(models_directory) / f"{safe_name or 'promoted_model'}.joblib"
