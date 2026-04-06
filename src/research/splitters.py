"""Time-series split helpers for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schemas import ResearchSplit


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Generate contiguous walk-forward splits over row indices."""

    train_size: int
    validation_size: int
    test_size: int
    step_size: int | None = None
    expanding_window: bool = True

    def split(self, total_rows: int) -> List[ResearchSplit]:
        """Return deterministic walk-forward splits for a dataset length."""
        if total_rows <= 0:
            return []

        train_size = int(self.train_size)
        validation_size = int(self.validation_size)
        test_size = int(self.test_size)
        step_size = int(self.step_size or test_size)
        window_size = train_size + validation_size + test_size
        if min(train_size, validation_size, test_size, step_size) <= 0:
            raise ValueError("Walk-forward split sizes must be positive integers")
        if total_rows < window_size:
            return []

        splits: List[ResearchSplit] = []
        split_id = 1
        start = 0
        while start + window_size <= total_rows:
            train_start = 0 if self.expanding_window else start
            train_end = train_start + train_size
            if self.expanding_window:
                train_end = start + train_size
            validation_start = train_end
            validation_end = validation_start + validation_size
            test_start = validation_end
            test_end = test_start + test_size

            splits.append(
                ResearchSplit(
                    name=f"fold_{split_id:02d}",
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            split_id += 1
            start += step_size

        return splits
