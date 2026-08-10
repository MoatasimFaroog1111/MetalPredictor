from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import pandas as pd
import pytest

from metal_predictor.local_archives import ContentAddressedArchiveCatalog
from metal_predictor.long_history import LongHistoryH1Stitcher
from metal_predictor.market_source import DownloadWindow
from metal_predictor.metastock_source import MetaStockM1ArchiveParser
from metal_predictor.stress_split import AnnualStressConfig, PurgedCalendarYearSplitter
from metal_predictor.stress_statistics import YearStratifiedCircularBlockBootstrap


def _write_metastock_zip(path: Path, rows: list[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("DAT_MS_XAGUSD_M1_TEST.csv", "\n".join(rows) + "\n")
        zf.writestr("DAT_MS_XAGUSD_M1_TEST.txt", "\n".join(rows) + "\n")


def test_metastock_parser_applies_fixed_est_to_utc(tmp_path: Path) -> None:
    archive = tmp_path / "HISTDATA_COM_MS_XAGUSD_M1202601.zip"
    _write_metastock_zip(
        archive,
        [
            "XAGUSD,202601010000,30.0,30.2,29.9,30.1,0",
            "XAGUSD,202601010001,30.1,30.3,30.0,30.2,0",
        ],
    )
    parsed = MetaStockM1ArchiveParser().parse((archive,))
    assert parsed["timestamp_utc"].iloc[0] == pd.Timestamp("2026-01-01T05:00:00Z")
    assert parsed["minute_valid_ohlc"].all()
    assert parsed["source_symbol"].eq("XAGUSD").all()


def test_metastock_parser_rejects_unexpected_symbol(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    _write_metastock_zip(
        archive,
        ["XAUUSD,202601010000,30.0,30.2,29.9,30.1,0"],
    )
    with pytest.raises(ValueError, match="symbols outside XAGUSD"):
        MetaStockM1ArchiveParser().parse((archive,))


def test_content_catalog_deduplicates_only_identical_zip_bytes(tmp_path: Path) -> None:
    original = tmp_path / "HISTDATA_COM_MS_XAGUSD_M12017.zip"
    copy = tmp_path / "HISTDATA_COM_MS_XAGUSD_M12017 (1).zip"
    different = tmp_path / "HISTDATA_COM_MS_XAGUSD_M12018.zip"
    _write_metastock_zip(
        original,
        ["XAGUSD,201701010000,10,11,9,10.5,0"],
    )
    shutil.copyfile(original, copy)
    _write_metastock_zip(
        different,
        ["XAGUSD,201801010000,11,12,10,11.5,0"],
    )
    selected, report = ContentAddressedArchiveCatalog().discover(tmp_path)
    assert len(selected) == 2
    assert report.discovered_files == 3
    assert report.unique_content_files == 2
    assert report.duplicate_content_files == 1
    assert original in selected
    assert copy not in selected


def _h1(timestamp: str, close: float, archive: str) -> dict[str, object]:
    return {
        "timestamp_utc": pd.Timestamp(timestamp),
        "open_usd_per_oz": close,
        "high_usd_per_oz": close,
        "low_usd_per_oz": close,
        "close_usd_per_oz": close,
        "open_usd_per_kg": close * 32.15074656862798,
        "high_usd_per_kg": close * 32.15074656862798,
        "low_usd_per_kg": close * 32.15074656862798,
        "close_usd_per_kg": close * 32.15074656862798,
        "open_value": close * 32.15074656862798,
        "high_value": close * 32.15074656862798,
        "low_value": close * 32.15074656862798,
        "close_value": close * 32.15074656862798,
        "minute_count": 60,
        "quality_flag": "OK",
        "asset": "XAG",
        "source_symbol": "XAGUSD",
        "source_provider": "HistData",
        "market_type": "spot_bid",
        "currency": "USD",
        "price_unit": "USD/kg",
        "source_archive": archive,
    }


def test_stitcher_excludes_conflicting_overlapping_h1_timestamp() -> None:
    first = pd.DataFrame(
        [
            _h1("2020-01-01T00:00:00Z", 10.0, "a.zip"),
            _h1("2020-01-01T01:00:00Z", 11.0, "a.zip"),
        ]
    )
    second = pd.DataFrame(
        [
            _h1("2020-01-01T01:00:00Z", 99.0, "b.zip"),
            _h1("2020-01-01T02:00:00Z", 12.0, "b.zip"),
        ]
    )
    output, overlaps, conflicts = LongHistoryH1Stitcher().stitch(
        (first, second),
        DownloadWindow(
            start_utc=pd.Timestamp("2020-01-01T00:00:00Z"),
            end_utc=pd.Timestamp("2020-01-01T02:00:00Z"),
        ),
    )
    assert overlaps == 1
    assert conflicts == 1
    assert output["timestamp_utc"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00Z"),
        pd.Timestamp("2020-01-01T02:00:00Z"),
    ]


def test_calendar_year_splitter_purges_future_target_labels() -> None:
    timestamps = pd.date_range("2010-01-01", "2014-12-31 23:00", freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "target_timestamp_utc": timestamps + pd.Timedelta(hours=1),
        }
    )
    splitter = PurgedCalendarYearSplitter(
        AnnualStressConfig(
            first_evaluation_year=2012,
            last_evaluation_year=2014,
            min_train_rows=5000,
        )
    )
    folds = splitter.split(frame)
    assert [fold.year for fold in folds] == [2012, 2013, 2014]
    for fold in folds:
        first_validation = fold.validation["timestamp_utc"].iloc[0]
        assert fold.train["target_timestamp_utc"].max() < first_validation
        assert fold.train["timestamp_utc"].max() < first_validation


def test_calendar_year_splitter_skips_undertrained_early_year_without_relaxing_minimum() -> None:
    timestamps = pd.date_range("2011-07-01", "2014-12-31 23:00", freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "target_timestamp_utc": timestamps + pd.Timedelta(hours=1),
        }
    )
    splitter = PurgedCalendarYearSplitter(
        AnnualStressConfig(
            first_evaluation_year=2012,
            last_evaluation_year=2014,
            min_train_rows=7000,
            skip_insufficient_train_years=True,
        )
    )
    folds = splitter.split(frame)
    assert [fold.year for fold in folds] == [2013, 2014]
    assert all(len(fold.train) >= 7000 for fold in folds)


def test_year_stratified_bootstrap_is_deterministic_and_keeps_positive_mean() -> None:
    frame = pd.DataFrame(
        {
            "year": [2020] * 100 + [2021] * 100,
            "edge": [0.01] * 200,
        }
    )
    bootstrap = YearStratifiedCircularBlockBootstrap(
        block_size_rows=24, resamples=500, random_state=42
    )
    first = bootstrap.sample_means(frame, ("edge",))
    second = bootstrap.sample_means(frame, ("edge",))
    pd.testing.assert_frame_equal(first, second)
    summary = bootstrap.summarize(frame["edge"], first["edge"], 0.0)
    assert summary.ci95_low == pytest.approx(0.01)
    assert summary.ci95_high == pytest.approx(0.01)
    assert summary.probability_above_threshold == 1.0
