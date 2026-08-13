from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    directional_accuracy: float
    row_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "mae": self.mae,
            "directional_accuracy": self.directional_accuracy,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class PairedBlockBootstrapResult:
    iterations: int
    block_length_rows: int
    seed: int
    ci_level: float
    mean_mae_improvement: float
    ci_low: float
    ci_high: float
    probability_candidate_mae_better: float
    method: str = "CIRCULAR_MOVING_BLOCK_BOOTSTRAP"

    def as_dict(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "block_length_rows": self.block_length_rows,
            "seed": self.seed,
            "ci_level": self.ci_level,
            "mean_mae_improvement": self.mean_mae_improvement,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "probability_candidate_mae_better": self.probability_candidate_mae_better,
            "method": self.method,
        }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if y.size == 0:
        raise ValueError("At least one observation is required.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    mae = float(np.mean(np.abs(y - p)))
    directional = float(np.mean(np.sign(y) == np.sign(p)))
    return RegressionMetrics(mae=mae, directional_accuracy=directional, row_count=int(y.size))


def paired_block_bootstrap_mae_improvement(
    actual: np.ndarray,
    baseline_predicted: np.ndarray,
    candidate_predicted: np.ndarray,
    *,
    iterations: int,
    block_length_rows: int,
    seed: int,
    ci_level: float,
) -> PairedBlockBootstrapResult:
    """Bootstrap paired per-row absolute-error improvements.

    Positive values mean the candidate has smaller absolute error than the random-walk
    baseline. Circular blocks preserve short-range ordering while avoiding fabricated
    observations or any access to the locked historical-test partition.
    """

    y = np.asarray(actual, dtype=float).reshape(-1)
    b = np.asarray(baseline_predicted, dtype=float).reshape(-1)
    c = np.asarray(candidate_predicted, dtype=float).reshape(-1)
    if y.shape != b.shape or y.shape != c.shape:
        raise ValueError("Paired bootstrap arrays must have identical shapes.")
    if y.size == 0:
        raise ValueError("Paired bootstrap requires observations.")
    if not np.isfinite(y).all() or not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("Paired bootstrap requires finite values.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if block_length_rows <= 0:
        raise ValueError("block_length_rows must be positive.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between zero and one.")

    improvement = np.abs(y - b) - np.abs(y - c)
    n = int(improvement.size)
    block_length = min(int(block_length_rows), n)
    block_count = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)
    rng = np.random.default_rng(int(seed))
    bootstrap_means = np.empty(int(iterations), dtype=float)

    for iteration in range(int(iterations)):
        starts = rng.integers(0, n, size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % n
        sample = improvement[indices.reshape(-1)[:n]]
        bootstrap_means[iteration] = float(np.mean(sample))

    alpha = 1.0 - float(ci_level)
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return PairedBlockBootstrapResult(
        iterations=int(iterations),
        block_length_rows=int(block_length_rows),
        seed=int(seed),
        ci_level=float(ci_level),
        mean_mae_improvement=float(np.mean(improvement)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        probability_candidate_mae_better=float(np.mean(bootstrap_means > 0.0)),
    )
