from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping


CONFIRMATION_VERSION = "precious-metals-historical-confirmation-v1"
CANDIDATE_ID = "xpt-xpd-candle-shape-own-returns-v1"
CANDIDATE_FAMILIES: tuple[str, ...] = ("candle_shape", "own_returns")
CANDIDATE_FEATURES: tuple[str, ...] = (
    "xpt_candle_range_pct",
    "xpt_candle_body_pct",
    "xpd_candle_range_pct",
    "xpd_candle_body_pct",
    "xpt_log_return_1h",
    "xpt_log_return_6h",
    "xpt_log_return_24h",
    "xpd_log_return_1h",
    "xpd_log_return_6h",
    "xpd_log_return_24h",
)


def candidate_fingerprint() -> str:
    payload = {
        "confirmation_version": CONFIRMATION_VERSION,
        "candidate_id": CANDIDATE_ID,
        "families": list(CANDIDATE_FAMILIES),
        "features": list(CANDIDATE_FEATURES),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HistoricalConfirmationPolicy:
    """One-shot historical Test confirmation gate fixed before Test is read.

    The old Test is not the formal Future Holdout. It may confirm or reject this exact
    already-selected candidate once, but must never be used to tune or replace features
    after the result is observed.
    """

    minimum_joint_coverage: float = 0.90
    minimum_mae_improvement: float = 0.0
    minimum_bootstrap_probability_better: float = 0.95
    minimum_ci95_low: float = 0.0
    minimum_directional_accuracy_delta: float = -0.01
    bootstrap_block_rows: int = 24
    bootstrap_resamples: int = 5000

    def __post_init__(self) -> None:
        if not 0 < self.minimum_joint_coverage <= 1:
            raise ValueError("minimum_joint_coverage must be in (0, 1].")
        if not 0.5 < self.minimum_bootstrap_probability_better <= 1:
            raise ValueError("minimum_bootstrap_probability_better must be in (0.5, 1].")
        if self.bootstrap_block_rows < 2 or self.bootstrap_resamples < 200:
            raise ValueError("Historical confirmation bootstrap configuration is invalid.")

    def decide(
        self,
        *,
        bootstrap: Mapping[str, object],
        base_metrics: Mapping[str, object],
        candidate_metrics: Mapping[str, object],
        joint_coverage: float,
    ) -> dict[str, object]:
        directional_delta = (
            float(candidate_metrics["directional_accuracy"])
            - float(base_metrics["directional_accuracy"])
        )
        checks = {
            "joint_coverage": joint_coverage >= self.minimum_joint_coverage,
            "mae_improvement_positive": (
                float(bootstrap["mae_improvement_vs_baseline"])
                > self.minimum_mae_improvement
            ),
            "bootstrap_probability": (
                float(bootstrap["probability_selected_better"])
                >= self.minimum_bootstrap_probability_better
            ),
            "ci95_low_positive": (
                float(bootstrap["improvement_ci95_low"])
                > self.minimum_ci95_low
            ),
            "directional_accuracy_not_materially_worse": (
                directional_delta >= self.minimum_directional_accuracy_delta
            ),
        }
        passed = all(checks.values())
        return {
            "status": "CONFIRMED" if passed else "REJECTED_ON_HISTORICAL_TEST",
            "confirmed": passed,
            "checks": checks,
            "directional_accuracy_delta": directional_delta,
            "policy_fixed_before_old_test_read": True,
            "candidate_features_locked_before_old_test_read": True,
            "no_post_test_feature_tuning_allowed": True,
        }

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
