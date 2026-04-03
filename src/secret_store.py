"""
Local machine-only secret storage for broker credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def get_default_secret_store_path() -> Path:
    """Return the local machine path used for broker secrets."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "ai-gold-research-system" / "broker_secrets.json"
    return Path.home() / ".ai-gold-research-system" / "broker_secrets.json"


class BrokerSecretStore:
    """Persist broker secrets outside the tracked repository config."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else get_default_secret_store_path()

    def get_password(self, profile_name: str) -> str:
        payload = self._load()
        return str(payload.get("profiles", {}).get(profile_name, {}).get("password", ""))

    def has_password(self, profile_name: str) -> bool:
        return bool(self.get_password(profile_name).strip())

    def set_password(self, profile_name: str, password: str) -> None:
        payload = self._load()
        payload.setdefault("profiles", {})
        payload["profiles"].setdefault(profile_name, {})
        payload["profiles"][profile_name]["password"] = password
        self._save(payload)

    def delete_password(self, profile_name: str) -> None:
        payload = self._load()
        profiles = payload.get("profiles", {})
        if profile_name in profiles:
            del profiles[profile_name]
            self._save(payload)

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"profiles": {}}
        with self.path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj) or {"profiles": {}}

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
