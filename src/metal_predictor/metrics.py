from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class ForecastMetrics:
    rows: int
    mae_return: float
    rmse_return: float
    median_ae_return: float
    r2_return: float
    directional_accuracy: float
    pearson_ic: float | None
    spearman_ic: float | None
    mean_error_return: float
    prediction_std: float
    price_mae_usd_per_kg: float
    price_rmse_usd_per_kg: float
    price_mape_percent: float

    def as_dict(self):
        return asdict(self)


class RegressionForecastMetrics:
    """Pure metric component; never trains or selects models."""

    def calculate(self, actual_return, predicted_return, current_close, target_close) -> ForecastMetrics:
        y = np.asarray(actual_return, dtype=float)
        pred = np.asarray(predicted_return, dtype=float)
        current = np.asarray(current_close, dtype=float)
        target_price = np.asarray(target_close, dtype=float)
        if not (len(y) == len(pred) == len(current) == len(target_price)) or len(y) == 0:
            raise ValueError("Metric arrays must be non-empty and equally sized.")
        if not all(np.isfinite(a).all() for a in (y, pred, current, target_price)):
            raise ValueError("Metrics received NaN or infinite values.")
        predicted_price = current * np.exp(pred)
        actual_sign = np.sign(y)
        predicted_sign = np.sign(pred)
        nonzero = actual_sign != 0
        directional = float(np.mean(actual_sign[nonzero] == predicted_sign[nonzero])) if nonzero.any() else 0.0
        ape = np.abs((predicted_price - target_price) / target_price)
        return ForecastMetrics(
            rows=len(y),
            mae_return=float(mean_absolute_error(y, pred)),
            rmse_return=float(np.sqrt(mean_squared_error(y, pred))),
            median_ae_return=float(np.median(np.abs(y - pred))),
            r2_return=float(r2_score(y, pred)),
            directional_accuracy=directional,
            pearson_ic=self._safe_corr(y, pred),
            spearman_ic=self._safe_spearman(y, pred),
            mean_error_return=float(np.mean(pred - y)),
            prediction_std=float(np.std(pred)),
            price_mae_usd_per_kg=float(mean_absolute_error(target_price, predicted_price)),
            price_rmse_usd_per_kg=float(np.sqrt(mean_squared_error(target_price, predicted_price))),
            price_mape_percent=float(np.mean(ape) * 100.0),
        )

    @staticmethod
    def _safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        value = float(np.corrcoef(a, b)[0, 1])
        return value if np.isfinite(value) else None

    @staticmethod
    def _safe_spearman(a, b):
        return RegressionForecastMetrics._safe_corr(
            pd.Series(a).rank(method="average").to_numpy(float),
            pd.Series(b).rank(method="average").to_numpy(float),
        )


@dataclass(frozen=True)
class BootstrapComparison:
    selected_mae_return: float
    baseline_mae_return: float
    mae_improvement_vs_baseline: float
    mae_improvement_percent: float
    improvement_ci95_low: float
    improvement_ci95_high: float
    probability_selected_better: float
    block_size_rows: int
    resamples: int

    def as_dict(self):
        return asdict(self)


class PairedBlockBootstrapComparison:
    """Post-selection uncertainty estimate for test MAE improvement."""

    def __init__(self, block_size_rows: int = 24, resamples: int = 2000, random_state: int = 42) -> None:
        if block_size_rows < 2 or resamples < 200:
            raise ValueError("Invalid bootstrap configuration.")
        self._block = block_size_rows
        self._resamples = resamples
        self._seed = random_state

    def compare(self, actual, selected_prediction, baseline_prediction) -> BootstrapComparison:
        y = np.asarray(actual, dtype=float)
        selected = np.asarray(selected_prediction, dtype=float)
        baseline = np.asarray(baseline_prediction, dtype=float)
        if not (len(y) == len(selected) == len(baseline)) or len(y) == 0:
            raise ValueError("Bootstrap arrays must be non-empty and equally sized.")
        selected_error = np.abs(y - selected)
        baseline_error = np.abs(y - baseline)
        improvement = baseline_error - selected_error
        rng = np.random.default_rng(self._seed)
        n = len(improvement)
        block = min(self._block, n)
        starts = np.arange(0, n - block + 1)
        blocks_needed = int(np.ceil(n / block))
        boot_means = np.empty(self._resamples, dtype=float)
        for i in range(self._resamples):
            chosen = rng.choice(starts, size=blocks_needed, replace=True)
            sample = np.concatenate([improvement[s:s + block] for s in chosen])[:n]
            boot_means[i] = np.mean(sample)
        baseline_mae = float(np.mean(baseline_error))
        observed = float(np.mean(improvement))
        low, high = np.quantile(boot_means, [0.025, 0.975])
        return BootstrapComparison(
            selected_mae_return=float(np.mean(selected_error)),
            baseline_mae_return=baseline_mae,
            mae_improvement_vs_baseline=observed,
            mae_improvement_percent=(observed / baseline_mae * 100.0) if baseline_mae else 0.0,
            improvement_ci95_low=float(low),
            improvement_ci95_high=float(high),
            probability_selected_better=float(np.mean(boot_means > 0.0)),
            block_size_rows=block,
            resamples=self._resamples,
        )
