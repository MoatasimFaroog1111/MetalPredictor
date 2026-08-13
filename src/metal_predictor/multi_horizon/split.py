from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final


@dataclass(frozen=True)
class WalkForwardFold:
    fold_number: int
    train_start: int
    train_end_exclusive: int
    validation_start: int
    validation_end_exclusive: int
    purge_bars: int

    @property
    def train_row_count(self) -> int:
        return self.train_end_exclusive - self.train_start

    @property
    def validation_row_count(self) -> int:
        return self.validation_end_exclusive - self.validation_start


@dataclass(frozen=True)
class LockedHistoricalTest:
    start: int
    end_exclusive: int

    @property
    def row_count(self) -> int:
        return self.end_exclusive - self.start


@dataclass(frozen=True)
class WalkForwardPlan:
    total_rows: int
    development_end_exclusive: int
    folds: tuple[WalkForwardFold, ...]
    historical_test: LockedHistoricalTest


class ExpandingWalkForwardPlanner:
    """Deterministic expanding-window planner with a one-target-bar purge."""

    fold_count: Final = 4
    purge_bars: Final = 1
    final_test_fraction: Final = 0.20
    minimum_test_rows: Final = 30
    minimum_train_rows: Final = 60
    minimum_validation_rows: Final = 15

    def plan(self, total_rows: int) -> WalkForwardPlan:
        if total_rows <= 0:
            raise ValueError("total_rows must be positive.")

        test_rows = max(self.minimum_test_rows, int(math.ceil(total_rows * self.final_test_fraction)))
        development_end = total_rows - test_rows
        if development_end <= self.minimum_train_rows + self.purge_bars:
            raise ValueError("Dataset is too small for the preregistered development/test split.")

        validation_size = (
            development_end - self.minimum_train_rows - self.purge_bars
        ) // self.fold_count
        if validation_size < self.minimum_validation_rows:
            raise ValueError("Dataset is too small for four preregistered walk-forward folds.")

        first_validation_start = development_end - validation_size * self.fold_count
        folds: list[WalkForwardFold] = []
        for fold_index in range(self.fold_count):
            validation_start = first_validation_start + fold_index * validation_size
            validation_end = (
                development_end
                if fold_index == self.fold_count - 1
                else validation_start + validation_size
            )
            train_end = validation_start - self.purge_bars
            if train_end < self.minimum_train_rows:
                raise ValueError("Preregistered fold violates minimum training size.")
            folds.append(
                WalkForwardFold(
                    fold_number=fold_index + 1,
                    train_start=0,
                    train_end_exclusive=train_end,
                    validation_start=validation_start,
                    validation_end_exclusive=validation_end,
                    purge_bars=self.purge_bars,
                )
            )

        return WalkForwardPlan(
            total_rows=total_rows,
            development_end_exclusive=development_end,
            folds=tuple(folds),
            historical_test=LockedHistoricalTest(start=development_end, end_exclusive=total_rows),
        )
