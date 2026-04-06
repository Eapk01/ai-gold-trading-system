"""Trainer implementations and shared trainer contracts."""

from .base import ResearchTrainer
from .current_ensemble import CurrentEnsembleTrainer
from .registry import TrainerRegistry

__all__ = [
    "CurrentEnsembleTrainer",
    "ResearchTrainer",
    "TrainerRegistry",
]
