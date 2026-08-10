from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultipleTestResult:
    strategy: str
    mean_return_per_period: float
    raw_one_sided_p_value: float
    holm_adjusted_p_value: float
    reject_zero_mean_at_5pct: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BlockBootstrapHolmTester:
    """One-sided zero-mean tests with dependence-aware block bootstrap + Holm correction.

    A common sequence of resampled contiguous blocks is applied to all strategies,
    preserving cross-strategy dependence. Each strategy is centered under its
    zero-mean null. Raw p-values are then adjusted across the entire comparable
    strategy family using Holm's step-down family-wise-error procedure.
    """

    def __init__(
        self,
        block_size_rows: int = 24,
        resamples: int = 5000,
        random_state: int = 42,
    ) -> None:
        if block_size_rows < 2 or resamples < 1000:
            raise ValueError("Block bootstrap requires block_size_rows>=2 and resamples>=1000.")
        self._block = block_size_rows
        self._resamples = resamples
        self._seed = random_state

    def test(
        self,
        returns: pd.DataFrame,
        strategy_names: tuple[str, ...],
    ) -> dict[str, object]:
        matrix = returns.loc[:, strategy_names].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("Multiple-testing return matrix contains non-finite values.")
        n, m = matrix.shape
        if n < self._block * 10 or m < 2:
            raise ValueError("Insufficient matrix size for block-bootstrap multiple testing.")
        observed = matrix.mean(axis=0)
        centred = matrix - observed
        rng = np.random.default_rng(self._seed)
        block = min(self._block, n)
        bootstrap_means = np.empty((self._resamples, m), dtype=float)

        for iteration in range(self._resamples):
            positions: list[int] = []
            while len(positions) < n:
                start = int(rng.integers(0, max(1, n - block + 1)))
                positions.extend(range(start, min(start + block, n)))
            idx = np.asarray(positions[:n], dtype=int)
            bootstrap_means[iteration] = centred[idx].mean(axis=0)

        raw_p = (
            1.0
            + (bootstrap_means >= observed.reshape(1, -1)).sum(axis=0)
        ) / (self._resamples + 1.0)
        adjusted = self._holm_adjust(raw_p)
        results = tuple(
            MultipleTestResult(
                strategy=name,
                mean_return_per_period=float(observed[index]),
                raw_one_sided_p_value=float(raw_p[index]),
                holm_adjusted_p_value=float(adjusted[index]),
                reject_zero_mean_at_5pct=bool(adjusted[index] < 0.05 and observed[index] > 0),
            )
            for index, name in enumerate(strategy_names)
        )
        return {
            "status": "PASS",
            "null_hypothesis": "strategy mean pre-cost log return <= 0",
            "alternative": "strategy mean pre-cost log return > 0",
            "block_size_rows": block,
            "resamples": self._resamples,
            "family_size": m,
            "holm_familywise_alpha": 0.05,
            "significant_strategy_count": sum(row.reject_zero_mean_at_5pct for row in results),
            "results": [row.as_dict() for row in results],
        }

    @staticmethod
    def _holm_adjust(raw_p: np.ndarray) -> np.ndarray:
        m = len(raw_p)
        order = np.argsort(raw_p)
        sorted_p = raw_p[order]
        adjusted_sorted = np.empty(m, dtype=float)
        running = 0.0
        for rank, p_value in enumerate(sorted_p):
            multiplier = m - rank
            candidate = min(1.0, multiplier * float(p_value))
            running = max(running, candidate)
            adjusted_sorted[rank] = running
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = adjusted_sorted
        return adjusted
