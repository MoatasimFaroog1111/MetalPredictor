from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DeflatedSharpeResult:
    selected_strategy: str
    observations: int
    counted_trials: int
    comparable_strategy_count: int
    observed_sharpe_per_period: float
    trial_sharpe_mean: float
    trial_sharpe_std: float
    selection_bias_benchmark_sharpe: float
    return_skewness: float
    return_raw_kurtosis: float
    z_statistic: float
    deflated_sharpe_probability: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DeflatedSharpeEvaluator:
    """Bailey/López de Prado Deflated Sharpe Ratio with an explicit trial ledger.

    All Sharpe ratios are kept in per-observation units. The expected maximum under
    the zero-skill null uses the authors' extreme-value approximation:

        E[max SR] = mu + sigma * [(1-gamma) Phi^-1(1-1/N)
                                  + gamma Phi^-1(1-1/(N e))]

    with mu=0 for the null and sigma estimated from comparable strategy Sharpe
    dispersion. The DSR is the Probabilistic Sharpe Ratio against that selection-
    bias benchmark, adjusted for observed skewness and raw kurtosis.
    """

    def evaluate(
        self,
        returns: pd.DataFrame,
        strategy_names: tuple[str, ...],
        selected_strategy: str,
        counted_trials: int,
    ) -> DeflatedSharpeResult:
        if selected_strategy not in strategy_names:
            raise ValueError(f"Selected strategy not in comparable matrix: {selected_strategy}")
        if counted_trials < len(strategy_names):
            raise ValueError("counted_trials cannot be smaller than comparable strategy count.")
        matrix = returns.loc[:, strategy_names].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("DSR strategy return matrix contains non-finite values.")

        trial_sharpes = np.asarray([
            self._sharpe(matrix[:, index]) for index in range(matrix.shape[1])
        ], dtype=float)
        selected = returns[selected_strategy].to_numpy(float)
        observed_sr = self._sharpe(selected)
        trial_std = float(trial_sharpes.std(ddof=1))
        trial_mean = float(trial_sharpes.mean())
        expected_max = self._expected_max_zero_null(trial_std, counted_trials)
        skewness, raw_kurtosis = self._moments(selected)
        denominator_sq = (
            1.0
            - skewness * observed_sr
            + ((raw_kurtosis - 1.0) / 4.0) * observed_sr * observed_sr
        )
        if denominator_sq <= 0 or not np.isfinite(denominator_sq):
            raise ValueError("DSR non-normality adjustment is not positive and finite.")
        z = (
            (observed_sr - expected_max)
            * np.sqrt(len(selected) - 1.0)
            / np.sqrt(denominator_sq)
        )
        probability = float(NormalDist().cdf(float(z)))
        return DeflatedSharpeResult(
            selected_strategy=selected_strategy,
            observations=len(selected),
            counted_trials=counted_trials,
            comparable_strategy_count=len(strategy_names),
            observed_sharpe_per_period=observed_sr,
            trial_sharpe_mean=trial_mean,
            trial_sharpe_std=trial_std,
            selection_bias_benchmark_sharpe=expected_max,
            return_skewness=skewness,
            return_raw_kurtosis=raw_kurtosis,
            z_statistic=float(z),
            deflated_sharpe_probability=probability,
        )

    @staticmethod
    def _sharpe(values: np.ndarray) -> float:
        std = float(np.std(values, ddof=1))
        if std <= 1e-15:
            raise ValueError("Sharpe is undefined for a constant return series.")
        return float(np.mean(values) / std)

    @staticmethod
    def _moments(values: np.ndarray) -> tuple[float, float]:
        centred = values - values.mean()
        m2 = float(np.mean(centred ** 2))
        if m2 <= 1e-30:
            raise ValueError("Return variance is too small for DSR moments.")
        m3 = float(np.mean(centred ** 3))
        m4 = float(np.mean(centred ** 4))
        skewness = m3 / (m2 ** 1.5)
        raw_kurtosis = m4 / (m2 ** 2)
        return float(skewness), float(raw_kurtosis)

    @staticmethod
    def _expected_max_zero_null(trial_sharpe_std: float, n_trials: int) -> float:
        if n_trials <= 1 or trial_sharpe_std <= 0:
            return 0.0
        normal = NormalDist()
        first = normal.inv_cdf(1.0 - 1.0 / n_trials)
        second = normal.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
        max_z = (1.0 - _EULER_MASCHERONI) * first + _EULER_MASCHERONI * second
        return float(trial_sharpe_std * max_z)
