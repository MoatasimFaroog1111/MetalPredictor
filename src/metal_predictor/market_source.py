from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
import zipfile

import numpy as np
import pandas as pd


FIXED_EST = timezone(timedelta(hours=-5))


@dataclass(frozen=True)
class InstrumentSpec:
    asset: str
    pair: str
    source_symbol: str
    provider: str = "HistData"
    market_type: str = "spot_bid"


@dataclass(frozen=True)
class DownloadWindow:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start_utc)
        end = pd.Timestamp(self.end_utc)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("DownloadWindow timestamps must be timezone-aware.")
        if start >= end:
            raise ValueError("DownloadWindow start must precede end.")


class ArchiveDownloader(Protocol):
    def download(self, instrument: InstrumentSpec, window: DownloadWindow, output_dir: Path) -> tuple[Path, ...]: ...


class MinuteArchiveParser(Protocol):
    def parse(self, archives: tuple[Path, ...]) -> pd.DataFrame: ...


class HistDataArchiveDownloader:
    """HistData adapter. It only downloads archives; parsing/quality are separate components."""

    def download(self, instrument: InstrumentSpec, window: DownloadWindow, output_dir: Path) -> tuple[Path, ...]:
        from histdata import download_hist_data
        from histdata.api import Platform, TimeFrame

        output_dir.mkdir(parents=True, exist_ok=True)
        source_start = pd.Timestamp(window.start_utc).tz_convert("UTC") - pd.Timedelta(hours=5)
        source_end = pd.Timestamp(window.end_utc).tz_convert("UTC") - pd.Timedelta(hours=5)
        current_year = datetime.now(timezone.utc).year
        paths: list[Path] = []

        for year in range(source_start.year, source_end.year + 1):
            if year > current_year:
                raise ValueError(f"Cannot download future HistData year {year}.")
            if year < current_year:
                path = download_hist_data(
                    year=str(year), month=None, pair=instrument.pair.lower(),
                    platform=Platform.GENERIC_ASCII, time_frame=TimeFrame.ONE_MINUTE,
                    output_directory=str(output_dir), verbose=True,
                )
                paths.append(Path(path))
                continue

            first_month = source_start.month if year == source_start.year else 1
            last_month = source_end.month if year == source_end.year else 12
            for month in range(first_month, last_month + 1):
                path = download_hist_data(
                    year=str(year), month=str(month), pair=instrument.pair.lower(),
                    platform=Platform.GENERIC_ASCII, time_frame=TimeFrame.ONE_MINUTE,
                    output_directory=str(output_dir), verbose=True,
                )
                paths.append(Path(path))

        if not paths:
            raise RuntimeError("HistData downloader produced no archives.")
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Downloaded archive paths missing: {missing}")
        return tuple(sorted(paths))


class GenericAsciiM1Parser:
    """Parses HistData Generic ASCII M1 bars and applies the documented fixed EST -> UTC conversion."""

    _COLUMNS = ("source_timestamp", "open", "high", "low", "close", "volume")

    def parse(self, archives: tuple[Path, ...]) -> pd.DataFrame:
        if not archives:
            raise ValueError("No HistData archives supplied.")
        frames: list[pd.DataFrame] = []
        for archive_sequence, archive in enumerate(sorted(archives)):
            frames.append(self._parse_one(archive, archive_sequence))
        out = pd.concat(frames, ignore_index=True)
        if out.empty:
            raise ValueError("HistData archives contained no minute rows.")
        return out

    def _parse_one(self, archive: Path, archive_sequence: int) -> pd.DataFrame:
        if not archive.exists():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as zf:
            candidates = [
                name for name in zf.namelist()
                if not name.endswith("/")
                and "STATUS" not in name.upper()
                and Path(name).suffix.lower() in {".csv", ".txt"}
            ]
            if not candidates:
                raise ValueError(f"No Generic ASCII M1 file found in {archive.name}.")
            member = sorted(candidates, key=lambda name: ("M1" not in name.upper(), len(name), name))[0]
            with zf.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    sep=";",
                    header=None,
                    names=self._COLUMNS,
                    dtype={"source_timestamp": "string"},
                    engine="c",
                )

        frame["archive_name"] = archive.name
        frame["archive_sequence"] = archive_sequence
        frame["source_row_number"] = np.arange(len(frame), dtype=np.int64)
        source_naive = pd.to_datetime(
            frame["source_timestamp"], format="%Y%m%d %H%M%S", errors="coerce"
        )
        if source_naive.isna().any():
            bad = int(source_naive.isna().sum())
            raise ValueError(f"{archive.name} contains {bad} invalid source timestamps.")
        frame["timestamp_source_est"] = source_naive
        frame["timestamp_utc"] = source_naive.dt.tz_localize(FIXED_EST).dt.tz_convert("UTC")

        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        prices = frame[["open", "high", "low", "close"]]
        finite = np.isfinite(prices.to_numpy(float)).all(axis=1)
        positive = (prices > 0).all(axis=1)
        invariants = (
            (frame["high"] >= frame["low"])
            & (frame["high"] >= frame[["open", "close"]].max(axis=1))
            & (frame["low"] <= frame[["open", "close"]].min(axis=1))
        )
        frame["minute_valid_ohlc"] = finite & positive & invariants
        return frame
