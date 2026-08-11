from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.live.inference import LivePredictionEngine


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_live_models_accept_expected_exact_clock_gap_nans() -> None:
    """A market closure creates expected NaN exact-lag features, not invalid input.

    The sealed Ridge payloads must reproduce the training-time median imputer and
    missing indicators. Live inference must not forward-fill or reject those NaNs.
    """
    engine = LivePredictionEngine(ROOT)
    timestamp = engine.historical_last_datetime_utc + timedelta(hours=48)
    close = 2500.0
    bar = HourlySilverBar(
        timestamp_utc=timestamp,
        open_usd_per_kg=close,
        high_usd_per_kg=close + 2.0,
        low_usd_per_kg=close - 2.0,
        close_usd_per_kg=close + 0.5,
        minute_count=60,
        quality_flag="OK",
        source_provider="SyntheticTest",
        source_symbol="XAGUSD",
        market_type="test_fixture",
    )

    snapshot = engine.predict([bar])

    assert snapshot.baseline_model == "ridge_alpha_100"
    assert snapshot.challenger_model == "ridge_alpha_10"
    assert np.isfinite(snapshot.baseline_log_return_1h)
    assert np.isfinite(snapshot.challenger_log_return_1h)
    assert snapshot.research_only is True
