from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from metal_predictor.direct_horizon.dataset import Stage7HorizonDataset


@dataclass(frozen=True)
class Stage7Fold:
    fold_number: int
    train_end_exclusive: int
    validation_start: int
    validation_end_exclusive: int
    validation_start_timestamp_utc: pd.Timestamp
    purge_hours: int
    embargo_hours: int

    @property
    def train_row_count(self) -> int:
        return self.train_end_exclusive

    @property
    def validation_row_count(self) -> int:
        return self.validation_end_exclusive - self.validation_start

    @property
    def purged_rows_before_validation(self) -> int:
        return self.validation_start - self.train_end_exclusive


@dataclass(frozen=True)
class Stage7SplitPlan:
    development_end_exclusive: int
    historical_test_start: int
    total_rows: int
    historical_test_boundary_utc: pd.Timestamp
    folds: tuple[Stage7Fold, ...]
    purge_hours: int
    embargo_hours: int

    @property
    def development_rows(self) -> int:
        return self.development_end_exclusive

    @property
    def historical_test_rows(self) -> int:
        return self.total_rows - self.historical_test_start


class Stage7PurgedExpandingPlanner:
    """Chronological expanding CV with horizon-sized label purge and structural embargo."""

    def __init__(
        self,
        *,
        historical_test_fraction: float = 0.20,
        initial_train_fraction: float = 0.50,
        fold_count: int = 4,
        minimum_train_rows: int = 5000,
    ) -> None:
        self._test_fraction = float(historical_test_fraction)
        self._initial_fraction = float(initial_train_fraction)
        self._fold_count = int(fold_count)
        self._minimum_train_rows = int(minimum_train_rows)
        if not 0.0 < self._test_fraction < 0.5:
            raise ValueError("historical_test_fraction must be between 0 and 0.5.")
        if not 0.2 <= self._initial_fraction < 0.9:
            raise ValueError("initial_train_fraction is outside the allowed range.")
        if self._fold_count < 2:
            raise ValueError("fold_count must be at least 2.")

    def plan(self, dataset: Stage7HorizonDataset) -> Stage7SplitPlan:
        frame = dataset.frame
        n = len(frame)
        if n < max(1000, self._minimum_train_rows * 2):
            raise ValueError("Too few Stage-7 rows for the preregistered split protocol.")
        ts = pd.to_datetime(frame[dataset.timestamp_name], utc=True)
        target_ts = pd.to_datetime(frame[dataset.target_timestamp_name], utc=True)

        raw_test_start = int(n * (1.0 - self._test_fraction))
        raw_test_start = min(max(raw_test_start, 1), n - 1)
        test_boundary = pd.Timestamp(ts.iloc[raw_test_start])

        # Development labels must terminate strictly before the locked test feature boundary.
        development_end = int((target_ts < test_boundary).sum())
        historical_test_start = int((ts < test_boundary).sum())
        if development_end <= 0 or historical_test_start >= n:
            raise ValueError("Invalid Stage-7 historical-test boundary.")
        if development_end > historical_test_start:
            raise ValueError("Development labels cross the historical-test feature boundary.")

        development = frame.iloc[:development_end]
        dev_n = len(development)
        initial_validation_start = max(
            self._minimum_train_rows,
            int(dev_n * self._initial_fraction),
        )
        if initial_validation_start >= dev_n - self._fold_count:
            raise ValueError("Not enough development rows for four validation folds.")
        remaining = dev_n - initial_validation_start
        base_size = remaining // self._fold_count
        if base_size <= 0:
            raise ValueError("Stage-7 validation fold size is zero.")

        folds: list[Stage7Fold] = []
        for i in range(self._fold_count):
            validation_start = initial_validation_start + i * base_size
            validation_end = (
                dev_n if i == self._fold_count - 1 else initial_validation_start + (i + 1) * base_size
            )
            validation_start_ts = pd.Timestamp(ts.iloc[validation_start])
            train_end = int((target_ts.iloc[:validation_start] < validation_start_ts).sum())
            if train_end < self._minimum_train_rows:
                raise ValueError(
                    f"Fold {i + 1} has only {train_end} rows after horizon purge."
                )
            if train_end > validation_start:
                raise ValueError("Purged training boundary crossed into validation.")
            # No future rows are ever admitted to Train in an expanding design, so the
            # post-validation embargo is structurally satisfied. We still record the
            # horizon-sized embargo explicitly for auditability.
            folds.append(
                Stage7Fold(
                    fold_number=i + 1,
                    train_end_exclusive=train_end,
                    validation_start=validation_start,
                    validation_end_exclusive=validation_end,
                    validation_start_timestamp_utc=validation_start_ts,
                    purge_hours=dataset.horizon.hours,
                    embargo_hours=dataset.horizon.hours,
                )
            )

        return Stage7SplitPlan(
            development_end_exclusive=development_end,
            historical_test_start=historical_test_start,
            total_rows=n,
            historical_test_boundary_utc=test_boundary,
            folds=tuple(folds),
            purge_hours=dataset.horizon.hours,
            embargo_hours=dataset.horizon.hours,
        )
