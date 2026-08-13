from __future__ import annotations

from datetime import timedelta
from typing import Final

from metal_predictor.forward_bars.admission import ForwardBarAdmissionPolicy
from metal_predictor.forward_bars.contracts import FORWARD_HORIZON_SECONDS, ForwardBar
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository


FORECAST_POLICY_VERSION: Final = "bullionvault-multi-horizon-baseline-forecast-v1"
BASELINE_ID: Final = "random_walk_zero_return"
STAGE3_SELECTION_LOCK_SHA256: Final = (
    "279b824c0775710b9b60a03a39564519e5ed728a54caca4febcefd71a24586f9"
)
_STAGE3_SELECTED_BASELINE: Final = frozenset({"4h", "12h", "2d", "30d"})


class MultiHorizonBaselineForecastService:
    """Publish only the scientifically retained random-walk research baseline.

    The latest completed forward bar must pass the Stage-5 admission policy. A rejected
    latest bar cannot be replaced by an older admitted bar. The service computes no
    performance metric and never promotes a model or creates a trading signal.
    """

    def __init__(
        self,
        repository: SQLiteForwardBarRepository,
        admission_policy: ForwardBarAdmissionPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._admission = admission_policy or ForwardBarAdmissionPolicy()

    @property
    def admission_policy(self) -> ForwardBarAdmissionPolicy:
        return self._admission

    def status(self) -> dict[str, object]:
        return {
            "component": "bullionvault-multi-horizon-research-forecast",
            "forecast_policy_version": FORECAST_POLICY_VERSION,
            "forecast_method": BASELINE_ID,
            "admission_policy": self._admission.specification,
            "horizons": {
                key: self.forecast(key, include_numeric_forecast=False)
                for key in FORWARD_HORIZON_SECONDS
            },
            "safety": self._safety(),
        }

    def forecast(
        self,
        horizon_key: str,
        *,
        include_numeric_forecast: bool = True,
    ) -> dict[str, object]:
        key = horizon_key.strip().lower()
        if key not in FORWARD_HORIZON_SECONDS:
            raise ValueError(f"Unsupported forecast horizon: {horizon_key!r}.")

        bars = self._repository.history(key, limit=500)
        admitted_count = sum(self._admission.evaluate(bar).admitted for bar in bars)
        evidence = {
            "completed_forward_bar_count": len(bars),
            "admitted_forward_bar_count": int(admitted_count),
            "admission_policy": self._admission.specification,
        }
        if not bars:
            return {
                "horizon_key": key,
                "interval_seconds": FORWARD_HORIZON_SECONDS[key],
                "state": "COLLECTING_EVIDENCE",
                "forecast_available": False,
                "reason": "NO_COMPLETED_FORWARD_BAR",
                "forecast_method": BASELINE_ID,
                "model_selection_evidence": self._selection_evidence(key),
                "evidence": evidence,
                "latest_bar": None,
                "safety": self._safety(),
            }

        latest = bars[-1]
        decision = self._admission.evaluate(latest)
        evidence["latest_bar_admission"] = decision.as_dict()
        evidence["latest_bar_quality"] = self._bar_quality(latest)
        if not decision.admitted:
            return {
                "horizon_key": key,
                "interval_seconds": latest.interval_seconds,
                "state": "COLLECTING_EVIDENCE",
                "forecast_available": False,
                "reason": "LATEST_COMPLETED_BAR_FAILED_ADMISSION_GATE",
                "forecast_method": BASELINE_ID,
                "model_selection_evidence": self._selection_evidence(key),
                "evidence": evidence,
                "latest_bar": self._bar_reference(latest),
                "safety": self._safety(),
            }

        target_start = latest.bucket_end_utc
        target_end = target_start + timedelta(seconds=latest.interval_seconds)
        response: dict[str, object] = {
            "horizon_key": key,
            "interval_seconds": latest.interval_seconds,
            "state": "BASELINE_FORECAST_AVAILABLE",
            "forecast_available": True,
            "reason": "LATEST_COMPLETED_BAR_ADMITTED",
            "forecast_method": BASELINE_ID,
            "forecast_semantics": "NEXT_FIXED_DURATION_FORWARD_BAR_CLOSE_MIDPOINT",
            "model_selection_evidence": self._selection_evidence(key),
            "reference": {
                "bar_start_utc": latest.bucket_start_utc.isoformat(),
                "bar_end_utc": latest.bucket_end_utc.isoformat(),
                "last_observed_snapshot_utc": latest.last_sample_at_utc.isoformat(),
                "close_mid_usd_per_kg": latest.close_mid_usd_per_kg,
                "close_bid_usd_per_kg": latest.close_bid_usd_per_kg,
                "close_ask_usd_per_kg": latest.close_ask_usd_per_kg,
                "source_provider": latest.source_provider,
                "source_stream": latest.source_stream,
            },
            "target": {
                "bar_start_utc": target_start.isoformat(),
                "bar_end_utc": target_end.isoformat(),
                "target_value_semantics": "NEXT_BAR_CLOSE_MIDPOINT",
            },
            "evidence": evidence,
            "safety": self._safety(),
        }
        if include_numeric_forecast:
            response["forecast"] = {
                "predicted_log_return": 0.0,
                "predicted_close_mid_usd_per_kg": latest.close_mid_usd_per_kg,
                "predicted_change_usd_per_kg": 0.0,
                "predicted_change_pct": 0.0,
                "direction_label": "FLAT_RANDOM_WALK_BASELINE",
                "prediction_interval_available": False,
                "prediction_interval_reason": "NO_CANDIDATE_PASSED_STAGE3_DEVELOPMENT_GATE",
            }
        return response

    @staticmethod
    def _selection_evidence(horizon_key: str) -> dict[str, object]:
        if horizon_key in _STAGE3_SELECTED_BASELINE:
            return {
                "selection_scope": "STAGE3_DEVELOPMENT_ONLY",
                "selected_id": BASELINE_ID,
                "selected_kind": "BASELINE",
                "candidate_gate_pass_count": 0,
                "historical_confirmation_authorized": False,
                "stage3_report_sha256": STAGE3_SELECTION_LOCK_SHA256,
            }
        return {
            "selection_scope": "BASELINE_REFERENCE_ONLY",
            "selected_id": BASELINE_ID,
            "selected_kind": "BASELINE",
            "candidate_gate_pass_count": None,
            "historical_confirmation_authorized": False,
            "note": "1d had no direct historical Stage-3 dataset; this is a reference baseline, not a selected predictive model.",
        }

    @staticmethod
    def _bar_quality(bar: ForwardBar) -> dict[str, object]:
        return {
            "snapshot_count": bar.snapshot_count,
            "expected_snapshot_count": bar.expected_snapshot_count,
            "coverage_ratio": bar.coverage_ratio,
            "quality_status": bar.quality_status,
            "access_mode_counts": dict(bar.access_mode_counts),
            "freshness_status_counts": dict(bar.freshness_status_counts),
        }

    @staticmethod
    def _bar_reference(bar: ForwardBar) -> dict[str, object]:
        return {
            "bar_start_utc": bar.bucket_start_utc.isoformat(),
            "bar_end_utc": bar.bucket_end_utc.isoformat(),
            "last_observed_snapshot_utc": bar.last_sample_at_utc.isoformat(),
            "close_mid_usd_per_kg": bar.close_mid_usd_per_kg,
            "quality": MultiHorizonBaselineForecastService._bar_quality(bar),
        }

    @staticmethod
    def _safety() -> dict[str, object]:
        return {
            "edge_status": "NOT_PROVEN",
            "research_only": True,
            "buy_sell_enabled": False,
            "execution_enabled": False,
            "automatic_promotion": False,
            "live_model_mutated": False,
            "frozen_52_feature_graph_mutated": False,
            "shadow62_mutated": False,
            "historical_chart_data_merged": False,
            "fill_or_interpolation_used": False,
            "performance_metrics_computed": False,
        }
