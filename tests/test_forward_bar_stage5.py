from __future__ import annotations

from datetime import datetime, timedelta, timezone

from metal_predictor.forward_bars.admission import ForwardBarAdmissionPolicy
from metal_predictor.forward_bars.contracts import FORWARD_HORIZON_SECONDS, ForwardBar
from metal_predictor.forward_bars.forecasting import MultiHorizonBaselineForecastService
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository


UTC = timezone.utc


def _bar(
    key: str,
    start: datetime,
    *,
    coverage: float = 0.95,
    access_mode: str = "AUTHENTICATED_READ_ONLY",
    freshness: str = "CURRENT_GUI_SOURCE",
    close: float = 2100.0,
) -> ForwardBar:
    interval = FORWARD_HORIZON_SECONDS[key]
    expected = max(2, interval // 60)
    count = max(2, min(expected, round(expected * coverage)))
    actual_coverage = count / expected
    end = start + timedelta(seconds=interval)
    return ForwardBar(
        horizon_key=key,
        interval_seconds=interval,
        bucket_start_utc=start,
        bucket_end_utc=end,
        source_provider="BullionVault",
        security_id="AGXLN",
        currency="USD",
        open_mid_usd_per_kg=2095.0,
        high_mid_usd_per_kg=2110.0,
        low_mid_usd_per_kg=2090.0,
        close_mid_usd_per_kg=close,
        open_bid_usd_per_kg=2094.0,
        close_bid_usd_per_kg=close - 1.0,
        open_ask_usd_per_kg=2096.0,
        close_ask_usd_per_kg=close + 1.0,
        mean_spread_usd_per_kg=2.0,
        max_spread_usd_per_kg=3.0,
        close_spread_usd_per_kg=2.0,
        snapshot_count=count,
        expected_snapshot_count=expected,
        coverage_ratio=actual_coverage,
        first_sample_at_utc=start + timedelta(minutes=1),
        last_sample_at_utc=end - timedelta(minutes=1),
        access_mode_counts={access_mode: count},
        freshness_status_counts={freshness: count},
        quality_status="TEST",
    )


def test_admission_requires_high_coverage_authenticated_current() -> None:
    policy = ForwardBarAdmissionPolicy()
    start = datetime(2026, 8, 13, 8, tzinfo=UTC)

    admitted = policy.evaluate(_bar("4h", start, coverage=0.95))
    partial = policy.evaluate(_bar("4h", start, coverage=0.52))
    public = policy.evaluate(
        _bar("4h", start, coverage=0.95, access_mode="PUBLIC_CACHED_READ_ONLY")
    )
    stale = policy.evaluate(
        _bar("4h", start, coverage=0.95, freshness="SERVER_CACHED_LESS_CURRENT")
    )

    assert admitted.admitted is True
    assert partial.admitted is False
    assert partial.reason == "REJECTED_INSUFFICIENT_COVERAGE"
    assert public.admitted is False
    assert stale.admitted is False
    assert policy.specification["historical_chart_fallback_allowed"] is False
    assert policy.specification["fill_allowed"] is False


def test_baseline_forecast_uses_latest_admitted_bar_without_claiming_edge(tmp_path) -> None:
    repository = SQLiteForwardBarRepository(tmp_path / "forward.sqlite3")
    start = datetime(2026, 8, 13, 8, tzinfo=UTC)
    bar = _bar("4h", start, coverage=0.95, close=2123.5)
    assert repository.append_bar(bar) is True

    payload = MultiHorizonBaselineForecastService(repository).forecast("4h")

    assert payload["state"] == "BASELINE_FORECAST_AVAILABLE"
    assert payload["forecast_available"] is True
    assert payload["forecast_method"] == "random_walk_zero_return"
    assert payload["forecast"]["predicted_log_return"] == 0.0
    assert payload["forecast"]["predicted_close_mid_usd_per_kg"] == 2123.5
    assert payload["target"]["bar_start_utc"] == bar.bucket_end_utc.isoformat()
    assert payload["target"]["bar_end_utc"] == (
        bar.bucket_end_utc + timedelta(hours=4)
    ).isoformat()
    assert payload["safety"]["edge_status"] == "NOT_PROVEN"
    assert payload["safety"]["execution_enabled"] is False
    assert payload["safety"]["performance_metrics_computed"] is False


def test_latest_rejected_bar_blocks_stale_fallback(tmp_path) -> None:
    repository = SQLiteForwardBarRepository(tmp_path / "forward.sqlite3")
    first = datetime(2026, 8, 13, 0, tzinfo=UTC)
    good = _bar("4h", first, coverage=0.96, close=2100.0)
    weak = _bar("4h", first + timedelta(hours=4), coverage=0.50, close=2110.0)
    assert repository.append_bar(good) is True
    assert repository.append_bar(weak) is True

    payload = MultiHorizonBaselineForecastService(repository).forecast("4h")

    assert payload["state"] == "COLLECTING_EVIDENCE"
    assert payload["forecast_available"] is False
    assert payload["reason"] == "LATEST_COMPLETED_BAR_FAILED_ADMISSION_GATE"
    assert payload["evidence"]["admitted_forward_bar_count"] == 1
    assert payload["evidence"]["latest_bar_admission"]["admitted"] is False
    assert "forecast" not in payload


def test_daily_is_reference_baseline_only_until_daily_evidence_is_admitted(tmp_path) -> None:
    repository = SQLiteForwardBarRepository(tmp_path / "forward.sqlite3")
    payload = MultiHorizonBaselineForecastService(repository).forecast("1d")

    assert payload["state"] == "COLLECTING_EVIDENCE"
    assert payload["model_selection_evidence"]["selection_scope"] == "BASELINE_REFERENCE_ONLY"
    assert payload["model_selection_evidence"]["candidate_gate_pass_count"] is None
    assert payload["model_selection_evidence"]["historical_confirmation_authorized"] is False


def test_status_never_exposes_numeric_forecast_values(tmp_path) -> None:
    repository = SQLiteForwardBarRepository(tmp_path / "forward.sqlite3")
    start = datetime(2026, 8, 13, 8, tzinfo=UTC)
    assert repository.append_bar(_bar("4h", start, coverage=0.99)) is True

    status = MultiHorizonBaselineForecastService(repository).status()

    assert status["horizons"]["4h"]["forecast_available"] is True
    assert "forecast" not in status["horizons"]["4h"]
    assert status["safety"]["buy_sell_enabled"] is False
    assert status["safety"]["historical_chart_data_merged"] is False
