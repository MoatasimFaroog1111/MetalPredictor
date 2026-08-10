from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapMeanSummary:
    observed_mean: float
    ci95_low: float
    ci95_high: float
    probability_above_threshold: float
    threshold: float
    block_size_rows: int
    resamples: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class YearStratifiedCircularBlockBootstrap:
    """Moving-block uncertainty while preserving each calendar year's weight.

    Blocks are sampled within a year and wrap circularly inside that year, so no
    bootstrap block can cross a year boundary. Multiple statistics can be sampled in
    one pass to avoid repeating expensive random-index construction.
    """

    def __init__(
        self,
        block_size_rows: int,
        resamples: int = 5000,
        random_state: int = 42,
    ) -> None:
        if block_size_rows < 2:
            raise ValueError("block_size_rows must be >= 2.")
        if resamples < 500:
            raise ValueError("resamples must be >= 500.")
        self._block = block_size_rows
        self._resamples = resamples
        self._seed = random_state

    def sample_means(
        self,
        frame: pd.DataFrame,
        value_columns: tuple[str, ...],
        year_column: str = "year",
    ) -> pd.DataFrame:
        if not value_columns:
            raise ValueError("At least one bootstrap value column is required.")
        missing = {year_column, *value_columns}.difference(frame.columns)
        if missing:
            raise ValueError(f"Bootstrap frame missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Bootstrap frame is empty.")

        rng = np.random.default_rng(self._seed)
        grouped = []
        total_rows = 0
        for _, group in frame.groupby(year_column, sort=True):
            values = group.loc[:, value_columns].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Bootstrap values must be finite.")
            n_rows = len(values)
            block = min(self._block, n_rows)
            blocks_needed = math.ceil(n_rows / block)
            final_rows = n_rows - (blocks_needed - 1) * block
            extended = np.vstack([values, values[:block]])
            cumulative = np.vstack(
                [np.zeros((1, len(value_columns))), np.cumsum(extended, axis=0)]
            )
            starts = np.arange(n_rows)
            full_sums = cumulative[starts + block] - cumulative[starts]
            final_sums = cumulative[starts + final_rows] - cumulative[starts]
            grouped.append((n_rows, blocks_needed, full_sums, final_sums))
            total_rows += n_rows

        sampled = np.empty((self._resamples, len(value_columns)), dtype=float)
        for sample_index in range(self._resamples):
            total = np.zeros(len(value_columns), dtype=float)
            for n_rows, blocks_needed, full_sums, final_sums in grouped:
                starts = rng.integers(0, n_rows, size=blocks_needed)
                if blocks_needed > 1:
                    total += full_sums[starts[:-1]].sum(axis=0)
                total += final_sums[starts[-1]]
            sampled[sample_index] = total / total_rows
        return pd.DataFrame(sampled, columns=value_columns)

    def summarize(
        self,
        observed: pd.Series,
        sampled: pd.Series,
        threshold: float,
    ) -> BootstrapMeanSummary:
        observed_values = pd.to_numeric(observed, errors="raise").to_numpy(float)
        sampled_values = pd.to_numeric(sampled, errors="raise").to_numpy(float)
        if not np.isfinite(observed_values).all() or not np.isfinite(sampled_values).all():
            raise ValueError("Bootstrap summary values must be finite.")
        low, high = np.quantile(sampled_values, [0.025, 0.975])
        return BootstrapMeanSummary(
            observed_mean=float(np.mean(observed_values)),
            ci95_low=float(low),
            ci95_high=float(high),
            probability_above_threshold=float(np.mean(sampled_values > threshold)),
            threshold=float(threshold),
            block_size_rows=self._block,
            resamples=self._resamples,
        )


def balanced_direction_accuracy(actual: np.ndarray, prediction: np.ndarray) -> float:
    y = np.sign(np.asarray(actual, dtype=float))
    pred = np.sign(np.asarray(prediction, dtype=float))
    nonzero = y != 0
    y = y[nonzero]
    pred = pred[nonzero]
    if not len(y):
        return 0.0
    positive = y > 0
    negative = y < 0
    sensitivity = float(np.mean(pred[positive] > 0)) if positive.any() else 0.0
    specificity = float(np.mean(pred[negative] < 0)) if negative.any() else 0.0
    return 0.5 * (sensitivity + specificity)


def position_turnover_units(prediction: np.ndarray, years: np.ndarray) -> tuple[float, float]:
    position = np.sign(np.asarray(prediction, dtype=float))
    year_values = np.asarray(years)
    if len(position) != len(year_values) or not len(position):
        raise ValueError("Position and year arrays must be non-empty and equally sized.")
    previous = np.r_[0.0, position[:-1]]
    year_start = np.r_[True, year_values[1:] != year_values[:-1]]
    turnover = np.abs(position - previous)
    turnover[year_start] = np.abs(position[year_start])
    flip_rate = float(np.mean(turnover == 2.0))
    return float(np.mean(turnover)), flip_rate
