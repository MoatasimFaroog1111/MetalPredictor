from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBOSplitResult:
    split: int
    in_sample_blocks: tuple[int, ...]
    selected_strategy: str
    selected_is_sharpe: float
    selected_oos_sharpe: float
    selected_oos_relative_rank: float
    rank_logit_lambda: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CSCVPBOEstimator:
    """Combinatorially Symmetric Cross-Validation estimate of backtest-overfitting probability.

    Implements the ranking logic described by Bailey, Borwein, López de Prado and
    Zhu, *The Probability of Backtest Overfitting* (Journal of Computational
    Finance, 2017). The backtest return matrix is split into an even number of
    contiguous blocks. For every half-block in-sample combination, the strategy with
    the highest in-sample Sharpe is selected and its out-of-sample rank is converted
    to a logit. PBO is the fraction of logits <= 0, i.e. selected winners landing in
    the lower half of the complementary OOS strategy ranking.
    """

    def __init__(self, n_blocks: int = 8) -> None:
        if n_blocks < 4 or n_blocks % 2:
            raise ValueError("CSCV requires an even n_blocks >= 4.")
        self._n_blocks = n_blocks

    def estimate(
        self,
        returns: pd.DataFrame,
        strategy_names: tuple[str, ...],
    ) -> dict[str, object]:
        if len(strategy_names) < 3:
            raise ValueError("PBO requires at least three comparable strategies.")
        matrix = returns.loc[:, strategy_names].to_numpy(float)
        if len(matrix) < self._n_blocks * 100:
            raise ValueError("Too few observations for requested CSCV block count.")
        if not np.isfinite(matrix).all():
            raise ValueError("PBO return matrix contains non-finite values.")

        block_indices = tuple(
            np.asarray(block, dtype=int)
            for block in np.array_split(np.arange(len(matrix)), self._n_blocks)
        )
        all_blocks = set(range(self._n_blocks))
        results: list[PBOSplitResult] = []
        selected_counts = {name: 0 for name in strategy_names}

        for split_number, is_blocks in enumerate(
            combinations(range(self._n_blocks), self._n_blocks // 2), start=1
        ):
            oos_blocks = tuple(sorted(all_blocks.difference(is_blocks)))
            is_idx = np.concatenate([block_indices[index] for index in is_blocks])
            oos_idx = np.concatenate([block_indices[index] for index in oos_blocks])
            is_sharpes = self._sharpes(matrix[is_idx])
            oos_sharpes = self._sharpes(matrix[oos_idx])
            selected_index = int(np.argmax(is_sharpes))
            selected_name = strategy_names[selected_index]
            selected_counts[selected_name] += 1

            # Ascending rank: 1=worst, M=best. Average ties match the paper's
            # percentile-rank intent without arbitrary tie breaking.
            ranks = pd.Series(oos_sharpes).rank(method="average", ascending=True).to_numpy(float)
            relative_rank = float(ranks[selected_index] / (len(strategy_names) + 1.0))
            if not 0.0 < relative_rank < 1.0:
                raise AssertionError("PBO relative rank must be strictly between 0 and 1.")
            rank_logit = float(np.log(relative_rank / (1.0 - relative_rank)))
            results.append(PBOSplitResult(
                split=split_number,
                in_sample_blocks=tuple(is_blocks),
                selected_strategy=selected_name,
                selected_is_sharpe=float(is_sharpes[selected_index]),
                selected_oos_sharpe=float(oos_sharpes[selected_index]),
                selected_oos_relative_rank=relative_rank,
                rank_logit_lambda=rank_logit,
            ))

        lambdas = np.asarray([row.rank_logit_lambda for row in results], dtype=float)
        pbo = float((lambdas <= 0.0).mean())
        selected_is = np.asarray([row.selected_is_sharpe for row in results], dtype=float)
        selected_oos = np.asarray([row.selected_oos_sharpe for row in results], dtype=float)
        degradation = selected_oos - selected_is
        return {
            "status": "PASS",
            "method": "Combinatorially Symmetric Cross-Validation (CSCV)",
            "n_blocks": self._n_blocks,
            "combinations": len(results),
            "strategy_count": len(strategy_names),
            "pbo": pbo,
            "probability_selected_winner_oos_above_median": 1.0 - pbo,
            "median_rank_logit_lambda": float(np.median(lambdas)),
            "mean_selected_is_sharpe_per_period": float(selected_is.mean()),
            "mean_selected_oos_sharpe_per_period": float(selected_oos.mean()),
            "mean_sharpe_degradation": float(degradation.mean()),
            "selected_strategy_counts": selected_counts,
            "split_results": [row.as_dict() for row in results],
        }

    @staticmethod
    def _sharpes(matrix: np.ndarray) -> np.ndarray:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        output = np.full(matrix.shape[1], -np.inf, dtype=float)
        valid = std > 1e-15
        output[valid] = mean[valid] / std[valid]
        return output
