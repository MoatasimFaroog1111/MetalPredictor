from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re

import pandas as pd


_REQUIRED_COLUMNS = ("Date", "High (kg)", "Low (kg)", "Close (kg)")
_DATE_FORMAT = "%H:%M:%S %d-%b-%Y"
_EXPORT_FILENAME_RE = re.compile(r"^AGX-USD-(?P<interval>\d+)-(?P<stamp>\d{14})\.csv$")


@dataclass(frozen=True)
class CsvAuditReport:
    filename: str
    sha256: str
    byte_count: int
    row_count: int
    expected_interval_seconds: int
    observed_interval_seconds: tuple[int, ...]
    oldest_timestamp_text: str
    newest_timestamp_text: str
    flat_hlc_rows: int
    empty_columns: tuple[str, ...]
    timestamp_timezone_semantics: str
    newest_bucket_status: str

    @property
    def exact_interval_pass(self) -> bool:
        return self.observed_interval_seconds == (self.expected_interval_seconds,)


class BullionVaultChartCsvAuditor:
    """Fail-closed audit of raw BullionVault chart CSV exports.

    The source file is read as-is. The auditor does not resample, fill, interpolate,
    infer a timezone, or fabricate an Open value.
    """

    timestamp_timezone_semantics = "UNVERIFIED_EXPORT_CLIENT_TIMEZONE"

    def audit(self, path: str | Path, *, expected_interval_seconds: int) -> CsvAuditReport:
        source = Path(path)
        raw = source.read_bytes()
        frame = pd.read_csv(source)

        missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError(f"BullionVault CSV missing required columns: {missing}.")
        if frame.empty:
            raise ValueError("BullionVault CSV must contain at least one row.")

        timestamps = pd.to_datetime(frame["Date"], format=_DATE_FORMAT, errors="raise")
        if timestamps.duplicated().any():
            raise ValueError("BullionVault CSV contains duplicate timestamps.")
        if not timestamps.is_monotonic_decreasing:
            raise ValueError("BullionVault raw export must remain newest-first.")

        ascending = timestamps.iloc[::-1].reset_index(drop=True)
        deltas = ascending.diff().dropna().dt.total_seconds().astype(int)
        observed = tuple(sorted(set(int(value) for value in deltas.tolist())))
        if observed != (int(expected_interval_seconds),):
            raise ValueError(
                f"Expected exact {expected_interval_seconds}s bars, observed {observed}."
            )

        numeric = frame.loc[:, ["High (kg)", "Low (kg)", "Close (kg)"]].apply(
            pd.to_numeric, errors="raise"
        )
        values = numeric.to_numpy(dtype=float)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values.ravel()):
            raise ValueError("BullionVault HLC values must be finite and positive.")
        if (numeric["High (kg)"] < numeric["Low (kg)"]).any():
            raise ValueError("BullionVault CSV contains High below Low.")
        if (
            (numeric["Close (kg)"] > numeric["High (kg)"])
            | (numeric["Close (kg)"] < numeric["Low (kg)"])
        ).any():
            raise ValueError("BullionVault CSV contains Close outside High/Low.")

        flat = (
            (numeric["High (kg)"] == numeric["Low (kg)"])
            & (numeric["Low (kg)"] == numeric["Close (kg)"])
        )
        empty_columns = tuple(
            str(column) for column in frame.columns if bool(frame[column].isna().all())
        )

        return CsvAuditReport(
            filename=source.name,
            sha256=hashlib.sha256(raw).hexdigest(),
            byte_count=len(raw),
            row_count=int(len(frame)),
            expected_interval_seconds=int(expected_interval_seconds),
            observed_interval_seconds=observed,
            oldest_timestamp_text=ascending.iloc[0].isoformat(),
            newest_timestamp_text=ascending.iloc[-1].isoformat(),
            flat_hlc_rows=int(flat.sum()),
            empty_columns=empty_columns,
            timestamp_timezone_semantics=self.timestamp_timezone_semantics,
            newest_bucket_status=self._newest_bucket_status(
                source.name,
                newest=ascending.iloc[-1].to_pydatetime(),
                interval_seconds=int(expected_interval_seconds),
            ),
        )

    @staticmethod
    def _newest_bucket_status(
        filename: str,
        *,
        newest: datetime,
        interval_seconds: int,
    ) -> str:
        match = _EXPORT_FILENAME_RE.match(filename)
        if match is None:
            return "UNVERIFIED"
        if int(match.group("interval")) != int(interval_seconds):
            return "UNVERIFIED"
        exported = datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")
        elapsed = (exported - newest).total_seconds()
        if 0 <= elapsed < interval_seconds:
            return "POTENTIALLY_INCOMPLETE"
        return "UNVERIFIED"


@dataclass(frozen=True)
class LoadedBullionVaultDataset:
    frame: pd.DataFrame
    audit: CsvAuditReport
    potentially_incomplete_newest_excluded: bool


class BullionVaultChartCsvLoader:
    """Conservative historical loader for Stage-1/Stage-2 research.

    Timestamps stay timezone-naive because the chart export timezone is not verified.
    This loader never resamples and never fills source gaps.
    """

    def __init__(self, auditor: BullionVaultChartCsvAuditor | None = None) -> None:
        self._auditor = auditor or BullionVaultChartCsvAuditor()

    def load(
        self,
        path: str | Path,
        *,
        expected_interval_seconds: int,
        include_potentially_incomplete_newest: bool = False,
    ) -> LoadedBullionVaultDataset:
        audit = self._auditor.audit(
            path,
            expected_interval_seconds=expected_interval_seconds,
        )
        frame = pd.read_csv(
            path,
            usecols=list(_REQUIRED_COLUMNS),
        )
        frame["timestamp_source"] = pd.to_datetime(
            frame.pop("Date"),
            format=_DATE_FORMAT,
            errors="raise",
        )
        frame = frame.rename(
            columns={
                "High (kg)": "high_usd_per_kg",
                "Low (kg)": "low_usd_per_kg",
                "Close (kg)": "close_usd_per_kg",
            }
        ).sort_values("timestamp_source", kind="stable").reset_index(drop=True)

        excluded = False
        if (
            not include_potentially_incomplete_newest
            and audit.newest_bucket_status == "POTENTIALLY_INCOMPLETE"
        ):
            frame = frame.iloc[:-1].reset_index(drop=True)
            excluded = True

        return LoadedBullionVaultDataset(
            frame=frame,
            audit=audit,
            potentially_incomplete_newest_excluded=excluded,
        )
