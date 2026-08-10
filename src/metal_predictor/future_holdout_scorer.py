from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.append_only_ledger import CsvHashChainLedger
from metal_predictor.future_holdout_collector import (
    OBSERVATION_COLUMNS,
    PREDICTION_COLUMNS,
)


@dataclass(frozen=True)
class BootstrapInterval:
    observed: float
    ci95_low: float
    ci95_high: float
    probability_positive: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FutureHoldoutScorer:
    """One-shot scorer for the predeclared fixed future window.

    Before both the calendar lock and minimum exact-hour outcome count are satisfied,
    the scorer returns LOCKED without computing or exposing performance metrics.
    Once unlocked it scores the fixed [start, end) feature window exactly once.
    """

    def __init__(self, frozen_root: Path, ledger_root: Path) -> None:
        self._frozen_root = frozen_root
        self._ledger_root = ledger_root
        self._manifest = json.loads(
            (frozen_root / "forward_holdout/freeze_manifest.json").read_text(encoding="utf-8")
        )
        self._observations = CsvHashChainLedger(
            ledger_root / "forward_holdout/observations.csv",
            OBSERVATION_COLUMNS,
            "timestamp_utc",
        )
        self._predictions = CsvHashChainLedger(
            ledger_root / "forward_holdout/predictions.csv",
            PREDICTION_COLUMNS,
            "feature_timestamp_utc",
        )
        self._final_path = ledger_root / "forward_holdout/final_score.json"

    def score(self, now_utc: pd.Timestamp | None = None) -> dict[str, object]:
        now = pd.Timestamp(now_utc or datetime.now(timezone.utc)).tz_convert("UTC")
        unlock = pd.Timestamp(self._manifest["earliest_final_score_utc"])
        if self._final_path.exists():
            existing = json.loads(self._final_path.read_text(encoding="utf-8"))
            self._verify_final_hash(existing)
            return {
                "status": "ALREADY_SCORED",
                "final_score_sha256": existing["final_score_sha256"],
                "performance_metrics_recomputed": False,
            }
        if now < unlock:
            return {
                "status": "LOCKED_TIME",
                "earliest_final_score_utc": unlock.isoformat(),
                "performance_metrics_computed": False,
            }

        eligible = self._eligible_outcomes()
        minimum = int(self._manifest["minimum_exact_hour_outcomes"])
        if len(eligible) < minimum:
            return {
                "status": "LOCKED_INSUFFICIENT_OUTCOMES",
                "minimum_exact_hour_outcomes": minimum,
                "eligible_exact_hour_outcomes": int(len(eligible)),
                "performance_metrics_computed": False,
            }

        report = self._compute_final(eligible, now)
        report["final_score_sha256"] = self._final_hash(report)
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._final_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._verify_final_hash(report)
        return report

    def _eligible_outcomes(self) -> pd.DataFrame:
        obs = self._observations.read_verified()
        pred = self._predictions.read_verified()
        if obs.empty or pred.empty:
            return pd.DataFrame()
        obs = obs.copy()
        pred = pred.copy()
        obs["timestamp_utc"] = pd.to_datetime(obs["timestamp_utc"], utc=True)
        pred["feature_timestamp_utc"] = pd.to_datetime(
            pred["feature_timestamp_utc"], utc=True
        )
        for column in (
            "close_usd_per_kg",
        ):
            obs[column] = pd.to_numeric(obs[column], errors="raise")
        for column in (
            "primary_prediction_log_return_1h",
            "benchmark_prediction_log_return_1h",
        ):
            pred[column] = pd.to_numeric(pred[column], errors="raise")

        start = pd.Timestamp(self._manifest["holdout_first_bar_start_utc"])
        end = pd.Timestamp(
            self._manifest["holdout_last_feature_bar_start_exclusive_utc"]
        )
        current = obs.loc[
            obs["timestamp_utc"].ge(start)
            & obs["timestamp_utc"].lt(end)
            & obs["holdout_role"].eq("HOLDOUT")
        ].copy()
        current = current.loc[:, ["timestamp_utc", "close_usd_per_kg"]].rename(
            columns={"close_usd_per_kg": "current_close_usd_per_kg"}
        )
        next_obs = obs.loc[:, ["timestamp_utc", "close_usd_per_kg"]].copy()
        next_obs["feature_timestamp_utc"] = (
            next_obs["timestamp_utc"] - pd.Timedelta(hours=1)
        )
        next_obs = next_obs.rename(
            columns={"close_usd_per_kg": "next_close_usd_per_kg"}
        ).drop(columns=["timestamp_utc"])
        joined = current.rename(
            columns={"timestamp_utc": "feature_timestamp_utc"}
        ).merge(
            next_obs,
            on="feature_timestamp_utc",
            how="inner",
            validate="one_to_one",
        ).merge(
            pred.loc[:, [
                "feature_timestamp_utc",
                "primary_model_name",
                "primary_prediction_log_return_1h",
                "benchmark_model_name",
                "benchmark_prediction_log_return_1h",
            ]],
            on="feature_timestamp_utc",
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            return joined
        joined["actual_log_return_1h"] = np.log(
            joined["next_close_usd_per_kg"]
            / joined["current_close_usd_per_kg"]
        )
        values = joined.loc[:, [
            "actual_log_return_1h",
            "primary_prediction_log_return_1h",
            "benchmark_prediction_log_return_1h",
        ]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Future holdout eligible outcomes contain invalid values.")
        return joined.sort_values("feature_timestamp_utc").reset_index(drop=True)

    def _compute_final(
        self,
        eligible: pd.DataFrame,
        now: pd.Timestamp,
    ) -> dict[str, object]:
        actual = eligible["actual_log_return_1h"].to_numpy(float)
        primary = eligible["primary_prediction_log_return_1h"].to_numpy(float)
        benchmark = eligible["benchmark_prediction_log_return_1h"].to_numpy(float)
        primary_correct = (np.sign(primary) == np.sign(actual)).astype(float)
        primary_strategy = np.sign(primary) * actual
        primary_abs_error = np.abs(actual - primary)
        zero_abs_error = np.abs(actual)
        benchmark_abs_error = np.abs(actual - benchmark)

        direction = self._block_bootstrap_stat(primary_correct, np.mean)
        strategy = self._block_bootstrap_stat(primary_strategy, np.mean)
        improvement_zero = self._block_bootstrap_stat(
            zero_abs_error - primary_abs_error, np.mean
        )
        improvement_benchmark = self._block_bootstrap_stat(
            benchmark_abs_error - primary_abs_error, np.mean
        )
        positive_blocks = self._positive_30day_blocks(eligible, primary_strategy)
        gates = {
            "directional_accuracy_ci95_low_above_50pct": direction.ci95_low > 0.50,
            "pre_cost_strategy_mean_ci95_low_above_zero": strategy.ci95_low > 0.0,
            "mae_improvement_vs_zero_ci95_low_above_zero": improvement_zero.ci95_low > 0.0,
            "mae_improvement_vs_benchmark_ci95_low_above_zero": (
                improvement_benchmark.ci95_low > 0.0
            ),
            "positive_30day_blocks_at_least_4_of_6": positive_blocks["positive_blocks"] >= 4,
        }
        predictive_edge = all(gates.values())
        report: dict[str, object] = {
            "status": "FINAL_SCORED",
            "scored_at_utc": now.isoformat(),
            "freeze_id": self._manifest["freeze_id"],
            "window": {
                "feature_start_utc": self._manifest["holdout_first_bar_start_utc"],
                "feature_end_exclusive_utc": self._manifest[
                    "holdout_last_feature_bar_start_exclusive_utc"
                ],
                "eligible_exact_hour_outcomes": int(len(eligible)),
            },
            "primary_model": str(eligible["primary_model_name"].iloc[0]),
            "benchmark_model": str(eligible["benchmark_model_name"].iloc[0]),
            "directional_accuracy": direction.as_dict(),
            "pre_cost_strategy_mean_log_return_per_signal": strategy.as_dict(),
            "mae_log_return": {
                "primary": float(primary_abs_error.mean()),
                "zero_return": float(zero_abs_error.mean()),
                "benchmark": float(benchmark_abs_error.mean()),
                "improvement_vs_zero": improvement_zero.as_dict(),
                "improvement_vs_benchmark": improvement_benchmark.as_dict(),
            },
            "six_predeclared_30day_blocks": positive_blocks,
            "proof_gates": gates,
            "decision": {
                "future_predictive_edge_proven_pre_cost": predictive_edge,
                "net_trading_edge_proven": False,
                "reason_net_trading_edge_not_proven": (
                    "HistData spot_bid H1 does not provide executable ask/spread/slippage/fees. "
                    "This protocol can establish frozen future predictive evidence only."
                ),
                "baseline_replacement_allowed": predictive_edge,
            },
            "methodology": {
                "one_shot_fixed_window": True,
                "interim_performance_disclosure": False,
                "bootstrap_block_rows": int(self._manifest["final_score_rules"]["bootstrap_block_rows"]),
                "bootstrap_resamples": int(self._manifest["final_score_rules"]["bootstrap_resamples"]),
                "batch_future_holdout_not_live_execution": True,
            },
        }
        return report

    def _block_bootstrap_stat(self, values: np.ndarray, stat_fn) -> BootstrapInterval:
        observed = float(stat_fn(values))
        n = len(values)
        block = int(self._manifest["final_score_rules"]["bootstrap_block_rows"])
        resamples = int(self._manifest["final_score_rules"]["bootstrap_resamples"])
        rng = np.random.default_rng(int(self._manifest["final_score_rules"]["random_state"]))
        samples = np.empty(resamples, dtype=float)
        for iteration in range(resamples):
            positions: list[int] = []
            while len(positions) < n:
                start = int(rng.integers(0, max(1, n - block + 1)))
                positions.extend(range(start, min(start + block, n)))
            idx = np.asarray(positions[:n], dtype=int)
            samples[iteration] = float(stat_fn(values[idx]))
        return BootstrapInterval(
            observed=observed,
            ci95_low=float(np.quantile(samples, 0.025)),
            ci95_high=float(np.quantile(samples, 0.975)),
            probability_positive=float((samples > 0.0).mean()),
        )

    def _positive_30day_blocks(
        self,
        eligible: pd.DataFrame,
        primary_strategy: np.ndarray,
    ) -> dict[str, object]:
        start = pd.Timestamp(self._manifest["holdout_first_bar_start_utc"])
        ts = pd.to_datetime(eligible["feature_timestamp_utc"], utc=True)
        details = []
        positive = 0
        for block_number in range(6):
            block_start = start + pd.Timedelta(days=30 * block_number)
            block_end = start + pd.Timedelta(days=30 * (block_number + 1))
            mask = ts.ge(block_start) & ts.lt(block_end)
            rows = int(mask.sum())
            mean_return = (
                float(primary_strategy[mask.to_numpy()].mean()) if rows else None
            )
            is_positive = bool(mean_return is not None and mean_return > 0.0)
            positive += int(is_positive)
            details.append({
                "block": block_number + 1,
                "start_utc": block_start.isoformat(),
                "end_utc": block_end.isoformat(),
                "rows": rows,
                "mean_pre_cost_strategy_log_return": mean_return,
                "positive": is_positive,
            })
        return {
            "positive_blocks": positive,
            "required_positive_blocks": 4,
            "blocks": details,
        }

    @staticmethod
    def _final_hash(report: dict[str, object]) -> str:
        clean = dict(report)
        clean.pop("final_score_sha256", None)
        canonical = json.dumps(
            clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _verify_final_hash(self, report: dict[str, object]) -> None:
        expected = report.get("final_score_sha256")
        if expected != self._final_hash(report):
            raise ValueError("Final future-holdout score hash mismatch.")
