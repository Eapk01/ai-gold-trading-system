"""Preprocessing utilities for research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class ResearchPreprocessor:
    """Simple numeric preprocessing with train-fold statistics."""

    fill_values: Dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "ResearchPreprocessor":
        """Learn per-column fill values from a training frame."""
        numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
        self.fill_values = {
            column: float(value)
            for column, value in numeric_frame.median(numeric_only=True).items()
            if pd.notna(value)
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned fill values to a frame."""
        numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
        if self.fill_values:
            numeric_frame = numeric_frame.fillna(self.fill_values)
        return numeric_frame.fillna(0.0)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(frame).transform(frame)
