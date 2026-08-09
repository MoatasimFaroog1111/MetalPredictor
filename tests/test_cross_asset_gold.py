from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.cross_asset_features import GoldSilverCrossAssetFeatures
from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import DownloadWindow, GenericAsciiM1Parser, InstrumentSpec


C = ColumnConfig()


def test_histdata_parser_converts_fixed_est_to_utc(tmp_path: Path) -> None:
    archive = tmp_path / "DAT_ASCII_XAUUSD_M1_2021.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "DAT_ASCII_XAUUSD_M1_2021.csv",
            "20210103 180000;1900.0;1901.0;1899.0;1900.5;0\n",
        )
    frame = GenericAsciiM1Parser().parse((archive,))
    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2021-01-03 23:00:00", tz="UTC")
    assert bool(frame.loc[0, "minute_valid_ohlc"])


def test_aggregator_excludes_ambiguous_overfull_conflicting_hour() -> None:
    rows = []
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    source_row = 0
    for hour in range(2):
        for minute in range(60):
            ts = start + pd.Timedelta(hours=hour, minutes=minute)
            price = 1900.0 + hour + minute / 1000.0
            rows.append({
                "timestamp_utc": ts,
                "archive_sequence": 0,
                "source_row_number": source_row,
                "minute_valid_ohlc": True,
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.1,
            })
            source_row += 1
            if hour == 0 and minute == 10:
                rows.append({
                    "timestamp_utc": ts,
                    "archive_sequence": 0,
                    "source_row_number": source_row,
                    "minute_valid_ohlc": True,
                    "open": price + 2.0,
                    "high": price + 2.2,
                    "low": price + 1.8,
                    "close": price + 2.1,
                })
                source_row += 1
    minutes = pd.DataFrame(rows)
    hourly, report = ConservativeH1Aggregator().aggregate(
        minutes,
        InstrumentSpec(asset="XAU", pair="xauusd", source_symbol="XAUUSD"),
        DownloadWindow(start, start + pd.Timedelta(hours=1)),
    )
    assert len(hourly) == 1
    assert hourly.loc[0, "timestamp_utc"] == start + pd.Timedelta(hours=1)
    assert report.conflicting_duplicate_timestamps == 1
    assert report.raw_hours_over_60_rows == 1
    assert report.excluded_suspicious_hours == 1


def test_exact_aligner_never_forward_fills_missing_auxiliary_hour() -> None:
    base = pd.Series(pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"))
    auxiliary = pd.DataFrame({
        "timestamp_utc": [base.iloc[0], base.iloc[2]],
        "value": [10.0, 30.0],
    })
    aligned = ExactTimestampAligner().align(base, auxiliary, ("value",), prefix="gold_")
    assert aligned.loc[0, "gold_value"] == 10.0
    assert np.isnan(aligned.loc[1, "gold_value"])
    assert aligned.loc[2, "gold_value"] == 30.0


def _silver_frame(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 1000.0 + np.arange(rows) * 0.05 + np.sin(np.arange(rows) / 15.0)
    return pd.DataFrame({
        C.timestamp: ts,
        C.open: close - 0.1,
        C.high: close + 0.3,
        C.low: close - 0.3,
        C.close: close,
        C.quality: "OK",
    })


def _gold_frame(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 60000.0 + np.arange(rows) * 1.5 + np.sin(np.arange(rows) / 11.0) * 10.0
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open_usd_per_kg": close - 2.0,
        "high_usd_per_kg": close + 4.0,
        "low_usd_per_kg": close - 4.0,
        "close_usd_per_kg": close,
        "quality_flag": "OK",
    })


def test_gold_cross_asset_features_are_causal_under_future_gold_perturbation() -> None:
    silver = _silver_frame()
    gold = _gold_frame()
    cutoff = 300

    baseline_component = GoldSilverCrossAssetFeatures(gold, ExactTimestampAligner(), C)
    baseline = baseline_component.transform(silver)

    perturbed_gold = gold.copy()
    price_cols = ["open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"]
    perturbed_gold.loc[cutoff + 1 :, price_cols] *= 5.0
    changed_component = GoldSilverCrossAssetFeatures(perturbed_gold, ExactTimestampAligner(), C)
    changed = changed_component.transform(silver)

    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff, baseline_component.feature_names],
        changed.loc[:cutoff, changed_component.feature_names],
        check_dtype=False,
        rtol=0.0,
        atol=1e-14,
    )


def test_missing_gold_bar_is_explicit_and_does_not_get_filled() -> None:
    silver = _silver_frame(50)
    gold = _gold_frame(50).drop(index=[20]).reset_index(drop=True)
    component = GoldSilverCrossAssetFeatures(gold, ExactTimestampAligner(), C, lags=(1, 3), windows=(24,))
    out = component.transform(silver)
    assert out.loc[20, "gold_has_exact_current"] == 0
    assert np.isnan(out.loc[20, "log_gold_silver_ratio"])
    assert np.isnan(out.loc[20, "gold_log_return_1h"])
