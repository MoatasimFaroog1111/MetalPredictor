from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnnualStressConfig:
    first_evaluation_year: int = 2012
    last_evaluation_year: int = 2026
    min_train_rows: int = 5000
    timestamp_name: str = "timestamp_utc"
    target_timestamp_name: str = "target_timestamp_utc"

    def __post_init__(self) -> None:
        if self.first_evaluation_year >= self.last_evaluation_year:
            raise ValueError("first_evaluation_year must precede last_evaluation_year.")
        if self.min_train_rows < 500:
            raise ValueError("min_train_rows must be >= 500.")


@dataclass(frozen=True)
class AnnualStressFold:
    year: int
    train: pd.DataFrame
    validation: pd.DataFrame


class PurgedCalendarYearSplitter:
    """Expanding, calendar-year validation with timestamp-aware label purging."""

    def __init__(self, config: AnnualStressConfig | None = None) -> None:
        self._config = config or AnnualStressConfig()

    def split(self, frame: pd.DataFrame) -> tuple[AnnualStressFold, ...]:
        c = self._config
        required = {c.timestamp_name, c.target_timestamp_name}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Stress frame missing split columns: {sorted(missing)}")
        ordered = frame.copy(deep=True)
        ordered[c.timestamp_name] = pd.to_datetime(
            ordered[c.timestamp_name], utc=True, errors="raise"
        )
        ordered[c.target_timestamp_name] = pd.to_datetime(
            ordered[c.target_timestamp_name], utc=True, errors="raise"
        )
        ordered = ordered.sort_values(c.timestamp_name).reset_index(drop=True)
        if ordered[c.timestamp_name].duplicated().any():
            raise ValueError("Stress frame timestamps must be unique.")

        folds: list[AnnualStressFold] = []
        for year in range(c.first_evaluation_year, c.last_evaluation_year + 1):
            year_start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
            year_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
            validation = ordered.loc[
                ordered[c.timestamp_name].ge(year_start)
                & ordered[c.timestamp_name].lt(year_end)
            ].copy()
            if validation.empty:
                continue
            first_validation = pd.Timestamp(validation[c.timestamp_name].iloc[0])
            train = ordered.loc[
                ordered[c.target_timestamp_name].lt(first_validation)
            ].copy()
            if len(train) < c.min_train_rows:
                raise ValueError(
                    f"Stress year {year} has only {len(train)} purged training rows."
                )
            if not pd.Timestamp(train[c.target_timestamp_name].max()) < first_validation:
                raise AssertionError(f"Stress year {year} failed target-time purge.")
            if not pd.Timestamp(train[c.timestamp_name].max()) < first_validation:
                raise AssertionError(f"Stress year {year} contains future feature rows.")
            folds.append(
                AnnualStressFold(
                    year=year,
                    train=train.reset_index(drop=True),
                    validation=validation.reset_index(drop=True),
                )
            )
        if not folds:
            raise ValueError("No calendar-year stress folds were produced.")
        return tuple(folds)
