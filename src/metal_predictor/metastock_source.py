from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from metal_predictor.market_source import FIXED_EST


class MetaStockM1ArchiveParser:
    """Parse HistData MetaStock M1 ZIPs into the canonical minute-frame contract.

    HistData MetaStock rows are expected as:
    SYMBOL,YYYYMMDDHHMM,OPEN,HIGH,LOW,CLOSE,VOLUME

    HistData documents this feed in fixed EST. The parser therefore applies the same
    fixed UTC-05:00 -> UTC policy already used by the project's Generic ASCII adapter.
    It never guesses daylight-saving offsets.
    """

    _COLUMNS = (
        "source_symbol",
        "source_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    _PRICE = ("open", "high", "low", "close")

    def __init__(self, expected_symbol: str = "XAGUSD") -> None:
        symbol = expected_symbol.strip().upper()
        if not symbol:
            raise ValueError("expected_symbol must not be empty.")
        self._expected_symbol = symbol

    def parse(self, archives: tuple[Path, ...]) -> pd.DataFrame:
        if not archives:
            raise ValueError("No MetaStock archives supplied.")
        frames = [self._parse_one(path, sequence) for sequence, path in enumerate(archives)]
        out = pd.concat(frames, ignore_index=True)
        if out.empty:
            raise ValueError("MetaStock archives contained no minute rows.")
        return out

    def _parse_one(self, archive: Path, archive_sequence: int) -> pd.DataFrame:
        if not archive.exists():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as zf:
            member = self._select_member(zf, archive)
            with zf.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    sep=",",
                    header=None,
                    names=self._COLUMNS,
                    dtype={
                        "source_symbol": "string",
                        "source_timestamp": "string",
                    },
                    engine="c",
                )

        if frame.empty:
            raise ValueError(f"{archive.name} contains no MetaStock rows.")
        normalized_symbol = frame["source_symbol"].str.upper().str.strip()
        unexpected = normalized_symbol.ne(self._expected_symbol)
        if unexpected.any():
            values = sorted(normalized_symbol.loc[unexpected].dropna().unique().tolist())
            raise ValueError(
                f"{archive.name} contains symbols outside {self._expected_symbol}: {values[:5]}"
            )
        frame["source_symbol"] = normalized_symbol
        frame["archive_name"] = archive.name
        frame["archive_sequence"] = archive_sequence
        frame["source_row_number"] = np.arange(len(frame), dtype=np.int64)

        source_naive = pd.to_datetime(
            frame["source_timestamp"],
            format="%Y%m%d%H%M",
            errors="coerce",
        )
        if source_naive.isna().any():
            raise ValueError(
                f"{archive.name} contains {int(source_naive.isna().sum())} invalid timestamps."
            )
        frame["timestamp_source_est"] = source_naive
        frame["timestamp_utc"] = source_naive.dt.tz_localize(FIXED_EST).dt.tz_convert("UTC")

        for column in (*self._PRICE, "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["minute_valid_ohlc"] = self._valid_ohlc_mask(frame)
        return frame

    @staticmethod
    def _select_member(zf: zipfile.ZipFile, archive: Path) -> str:
        files = [name for name in zf.namelist() if not name.endswith("/")]
        csv_files = [name for name in files if Path(name).suffix.lower() == ".csv"]
        candidates = csv_files or [
            name for name in files if Path(name).suffix.lower() == ".txt"
        ]
        if not candidates:
            raise ValueError(f"No MetaStock CSV/TXT member found in {archive.name}.")
        # HistData often ships equivalent CSV and TXT members. Prefer exactly one CSV.
        return sorted(candidates, key=lambda name: (len(name), name))[0]

    @classmethod
    def _valid_ohlc_mask(cls, frame: pd.DataFrame) -> pd.Series:
        prices = frame.loc[:, cls._PRICE]
        finite = np.isfinite(prices.to_numpy(float)).all(axis=1)
        positive = prices.gt(0).all(axis=1)
        invariants = (
            frame["high"].ge(frame["low"])
            & frame["high"].ge(frame[["open", "close"]].max(axis=1))
            & frame["low"].le(frame[["open", "close"]].min(axis=1))
        )
        return pd.Series(finite & positive & invariants, index=frame.index, dtype=bool)
