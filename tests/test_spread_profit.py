from __future__ import annotations

from datetime import datetime, timezone

import pytest

from metal_predictor.live.contracts import ForecastSnapshot
from metal_predictor.live.spread_profit import SpreadProfitAssumptions, SpreadProfitCalculator


def _forecast() -> ForecastSnapshot:
    return ForecastSnapshot(
        feature_timestamp_utc=datetime(2026, 8, 12, 6, 0, tzinfo=timezone.utc),
        decision_time_utc=datetime(2026, 8, 12, 7, 0, tzinfo=timezone.utc),
        current_price_usd_per_kg=2123.396057125035,
        baseline_model="ridge_alpha_100",
        baseline_log_return_1h=-0.0015910837403020734,
        baseline_predicted_price_usd_per_kg=2120.0202424983727,
        baseline_direction="DOWN",
        challenger_model="ridge_alpha_10",
        challenger_log_return_1h=-0.001822976643151077,
        challenger_predicted_price_usd_per_kg=2119.5286818471573,
        challenger_direction="DOWN",
        data_quality="PROVIDER_AGGREGATED_H1",
        source_provider="GoldAPI",
        source_compatible_with_training=False,
        materialized_at_utc=datetime(2026, 8, 12, 7, 5, 0, tzinfo=timezone.utc),
    )


def test_down_forecast_exposes_bid_ask_and_positive_short_margin() -> None:
    result = SpreadProfitCalculator().evaluate(
        _forecast(),
        SpreadProfitAssumptions(
            current_spread_usd_per_kg=1.0,
            forecast_spread_usd_per_kg=1.0,
            target_profit_usd=1.0,
            quantity_step_kg=0.001,
            minimum_trade_quantity_kg=0.001,
        ),
    )

    reference = result["reference"]
    baseline = result["baseline"]
    assert reference["current_bid_usd_per_kg"] == pytest.approx(2122.896057125035)
    assert reference["current_ask_usd_per_kg"] == pytest.approx(2123.896057125035)
    assert baseline["predicted_bid_usd_per_kg"] == pytest.approx(2119.5202424983727)
    assert baseline["predicted_ask_usd_per_kg"] == pytest.approx(2120.5202424983727)

    assert baseline["long"]["expected_margin_usd_per_kg"] < 0
    assert baseline["long"]["minimum_quantity_for_target_profit_kg"] is None
    assert baseline["short"]["expected_margin_usd_per_kg"] == pytest.approx(2.375814626662289)
    assert baseline["short"]["minimum_quantity_for_target_profit_kg"] == pytest.approx(0.421)
    assert baseline["short"]["expected_profit_at_minimum_quantity_usd"] >= 1.0

    assert result["assumptions"]["spread_source"] == "USER_ASSUMPTION"
    assert result["safety"]["buy_sell_enabled"] is False
    assert result["safety"]["execution_enabled"] is False


def test_wide_spread_can_remove_all_positive_margin() -> None:
    result = SpreadProfitCalculator().evaluate(
        _forecast(),
        SpreadProfitAssumptions(
            current_spread_usd_per_kg=10.0,
            forecast_spread_usd_per_kg=10.0,
            target_profit_usd=1.0,
        ),
    )

    for model_name in ("baseline", "challenger"):
        assert result[model_name]["long"]["minimum_quantity_for_target_profit_kg"] is None
        assert result[model_name]["short"]["minimum_quantity_for_target_profit_kg"] is None


def test_fixed_cost_is_included_in_minimum_quantity() -> None:
    result = SpreadProfitCalculator().evaluate(
        _forecast(),
        SpreadProfitAssumptions(
            current_spread_usd_per_kg=1.0,
            forecast_spread_usd_per_kg=1.0,
            target_profit_usd=5.0,
            fixed_round_trip_cost_usd=2.0,
            quantity_step_kg=0.001,
            minimum_trade_quantity_kg=0.001,
        ),
    )

    short = result["baseline"]["short"]
    assert short["minimum_quantity_for_target_profit_kg"] == pytest.approx(2.947)
    assert short["expected_profit_at_minimum_quantity_usd"] >= 5.0
