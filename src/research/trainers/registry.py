"""Registry for trainer implementations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import is_dataclass, replace
from dataclasses import dataclass, field
from typing import Dict

from .base import ResearchTrainer


@dataclass
class TrainerRegistry:
    """Simple in-memory registry for named trainer implementations."""

    trainers: Dict[str, ResearchTrainer] = field(default_factory=dict)

    def register(self, name: str, trainer: ResearchTrainer) -> None:
        """Register a trainer implementation under a stable name."""
        self.trainers[str(name).strip()] = trainer

    def get(self, name: str) -> ResearchTrainer:
        """Return a registered trainer by name."""
        key = str(name).strip()
        if key not in self.trainers:
            raise KeyError(f"Trainer is not registered: {key}")
        return self.trainers[key]

    def build(self, name: str, **overrides) -> ResearchTrainer:
        """Return a trainer instance with optional attribute overrides."""
        trainer = self.get(name)
        clean_overrides = {key: value for key, value in overrides.items() if value is not None}
        if not clean_overrides:
            return deepcopy(trainer)
        if is_dataclass(trainer):
            return replace(trainer, **clean_overrides)

        clone = deepcopy(trainer)
        for key, value in clean_overrides.items():
            if hasattr(clone, key):
                setattr(clone, key, value)
        return clone
