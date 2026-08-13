from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Protocol

from metal_predictor.precious_metals.confirmation import CANDIDATE_FEATURES, CANDIDATE_ID


SHADOW_PROTOCOL_VERSION = "xpt-xpd-shadow-holdout-v1"
SHADOW_FREEZE_ID = "xpt-xpd-shadow-62-v1-20260814"
SHADOW_FIRST_FEATURE_BAR_START_UTC = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC = datetime(2027, 2, 10, 0, 0, tzinfo=timezone.utc)
SHADOW_EARLIEST_FINAL_SCORE_UTC = datetime(2027, 2, 10, 2, 0, tzinfo=timezone.utc)
SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES = 2500
SHADOW_FIXED_WINDOW_DAYS = 180


@dataclass(frozen=True)
class ShadowForecastSnapshot:
    feature_timestamp_utc: datetime
    materialized_at_utc: datetime
    reference_close_usd_per_kg: float
    baseline_model: str
    baseline_model_sha256: str
    baseline_log_return_1h: float
    baseline_predicted_price_usd_per_kg: float
    candidate_id: str
    candidate_model: str
    candidate_model_sha256: str
    candidate_log_return_1h: float
    candidate_predicted_price_usd_per_kg: float
    xpt_exact_current: bool
    xpd_exact_current: bool
    auxiliary_provider: str

    def __post_init__(self) -> None:
        for name in ("feature_timestamp_utc", "materialized_at_utc"):
            value = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware.")
        if self.feature_timestamp_utc.minute or self.feature_timestamp_utc.second or self.feature_timestamp_utc.microsecond:
            raise ValueError("feature_timestamp_utc must align to an exact UTC hour.")
        values = (
            self.reference_close_usd_per_kg,
            self.baseline_log_return_1h,
            self.baseline_predicted_price_usd_per_kg,
            self.candidate_log_return_1h,
            self.candidate_predicted_price_usd_per_kg,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Shadow forecast contains non-finite numeric values.")
        if self.reference_close_usd_per_kg <= 0 or self.baseline_predicted_price_usd_per_kg <= 0 or self.candidate_predicted_price_usd_per_kg <= 0:
            raise ValueError("Shadow forecast prices must be positive.")
        if self.candidate_id != CANDIDATE_ID:
            raise ValueError("Shadow forecast candidate_id does not match the locked candidate.")

    @property
    def decision_time_utc(self) -> datetime:
        return self.feature_timestamp_utc + timedelta(hours=1)

    @property
    def target_bar_start_utc(self) -> datetime:
        return self.feature_timestamp_utc + timedelta(hours=1)

    @property
    def target_close_available_utc(self) -> datetime:
        return self.feature_timestamp_utc + timedelta(hours=2)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("feature_timestamp_utc", "materialized_at_utc"):
            data[key] = getattr(self, key).isoformat()
        data.update(
            {
                "decision_time_utc": self.decision_time_utc.isoformat(),
                "target_bar_start_utc": self.target_bar_start_utc.isoformat(),
                "target_close_available_utc": self.target_close_available_utc.isoformat(),
                "candidate_feature_count": len(CANDIDATE_FEATURES),
                "total_model_feature_count": 52 + len(CANDIDATE_FEATURES),
                "protocol_version": SHADOW_PROTOCOL_VERSION,
                "freeze_id": SHADOW_FREEZE_ID,
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
            }
        )
        return data


@dataclass(frozen=True)
class ShadowOutcome:
    target_bar_start_utc: datetime
    observed_at_utc: datetime
    actual_close_usd_per_kg: float
    source_provider: str
    quality_flag: str

    def __post_init__(self) -> None:
        if self.target_bar_start_utc.tzinfo is None or self.observed_at_utc.tzinfo is None:
            raise ValueError("Shadow outcome timestamps must be timezone-aware.")
        if self.target_bar_start_utc.minute or self.target_bar_start_utc.second or self.target_bar_start_utc.microsecond:
            raise ValueError("target_bar_start_utc must align to an exact UTC hour.")
        if not math.isfinite(float(self.actual_close_usd_per_kg)) or self.actual_close_usd_per_kg <= 0:
            raise ValueError("Shadow outcome close must be finite and positive.")

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["target_bar_start_utc"] = self.target_bar_start_utc.isoformat()
        data["observed_at_utc"] = self.observed_at_utc.isoformat()
        return data


class ShadowRepository(Protocol):
    def put_prediction(self, snapshot: ShadowForecastSnapshot) -> bool: ...
    def put_outcome(self, outcome: ShadowOutcome) -> bool: ...
    def latest_prediction(self) -> ShadowForecastSnapshot | None: ...
    def prediction_count(self) -> int: ...
    def outcome_count(self) -> int: ...
    def has_prediction(self, feature_timestamp_utc: datetime) -> bool: ...
    def has_prediction_for_target(self, target_bar_start_utc: datetime) -> bool: ...
