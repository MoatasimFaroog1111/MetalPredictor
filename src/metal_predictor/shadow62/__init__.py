"""Isolated 62-feature XPT/XPD shadow-candidate research runtime."""

from metal_predictor.shadow62.contracts import (
    SHADOW_EARLIEST_FINAL_SCORE_UTC,
    SHADOW_FIRST_FEATURE_BAR_START_UTC,
    SHADOW_FIXED_WINDOW_DAYS,
    SHADOW_FREEZE_ID,
    SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC,
    SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES,
    SHADOW_PROTOCOL_VERSION,
    ShadowForecastSnapshot,
    ShadowOutcome,
)
from metal_predictor.shadow62.engine import Shadow62InferenceEngine
from metal_predictor.shadow62.features import ConfirmedPreciousMetalsShadowFeatures
from metal_predictor.shadow62.repository import SQLiteShadowRepository
from metal_predictor.shadow62.scheduler import Shadow62Scheduler
from metal_predictor.shadow62.service import Shadow62Service

__all__ = [
    "SHADOW_PROTOCOL_VERSION",
    "SHADOW_FREEZE_ID",
    "SHADOW_FIRST_FEATURE_BAR_START_UTC",
    "SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC",
    "SHADOW_EARLIEST_FINAL_SCORE_UTC",
    "SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES",
    "SHADOW_FIXED_WINDOW_DAYS",
    "ShadowForecastSnapshot",
    "ShadowOutcome",
    "Shadow62InferenceEngine",
    "ConfirmedPreciousMetalsShadowFeatures",
    "SQLiteShadowRepository",
    "Shadow62Scheduler",
    "Shadow62Service",
]
