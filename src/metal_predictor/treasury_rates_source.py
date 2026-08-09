from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from metal_predictor.market_source import DownloadWindow
from metal_predictor.publication_time import H15PublicationPolicy


@dataclass(frozen=True)
class TreasuryRatesReport:
    rows: int
    first_observation_date: str
    last_observation_date: str
    missing_2y: int
    missing_10y: int
    delayed_h15_observations: int
    source: str
    current_vintage_warning: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TreasuryDailyParYieldCurveClient:
    """Downloads official U.S. Treasury daily par-yield CSVs, without API credentials."""

    _URL = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "daily-treasury-rates.csv/{year}/all?_format=csv&field_tdr_date_value={year}"
        "&page=&type=daily_treasury_yield_curve"
    )

    def __init__(self, publication_policy: H15PublicationPolicy | None = None) -> None:
        self._publication = publication_policy or H15PublicationPolicy()

    def fetch(self, window: DownloadWindow) -> tuple[pd.DataFrame, TreasuryRatesReport]:
        start = pd.Timestamp(window.start_utc).tz_convert("UTC").date()
        end = pd.Timestamp(window.end_utc).tz_convert("UTC").date()
        frames = [self._fetch_year(year) for year in range(start.year, end.year + 1)]
        raw = pd.concat(frames, ignore_index=True)
        raw = raw.loc[raw["observation_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        raw = raw.sort_values("observation_date").drop_duplicates("observation_date", keep="last").reset_index(drop=True)
        if raw.empty:
            raise ValueError("Treasury source returned no rows in requested window.")

        raw["available_from_utc"] = self._publication.available_from_utc(raw["observation_date"])
        raw["rate_2y_percent"] = pd.to_numeric(raw["rate_2y_percent"], errors="coerce")
        raw["rate_10y_percent"] = pd.to_numeric(raw["rate_10y_percent"], errors="coerce")
        finite_any = np.isfinite(raw[["rate_2y_percent", "rate_10y_percent"]].to_numpy(float)).any(axis=1)
        raw = raw.loc[finite_any].reset_index(drop=True)
        if raw["available_from_utc"].duplicated().any():
            raise ValueError("Multiple Treasury observations map to the same H.15 publication timestamp.")
        if not raw["available_from_utc"].is_monotonic_increasing:
            raise ValueError("Treasury publication timestamps are not chronological.")

        delayed = raw["observation_date"].dt.date.isin(self._publication._TREASURY_DELAY_OVERRIDES).sum()
        report = TreasuryRatesReport(
            rows=int(len(raw)),
            first_observation_date=str(raw["observation_date"].iloc[0].date()),
            last_observation_date=str(raw["observation_date"].iloc[-1].date()),
            missing_2y=int(raw["rate_2y_percent"].isna().sum()),
            missing_10y=int(raw["rate_10y_percent"].isna().sum()),
            delayed_h15_observations=int(delayed),
            source="U.S. Treasury Daily Treasury Par Yield Curve Rates",
            current_vintage_warning=(
                "Historical values are the Treasury's current official historical values. "
                "H.15 publication latency and documented 2023 omissions are modeled, but later "
                "historical corrections, if any, may already be incorporated."
            ),
        )
        return raw, report

    def _fetch_year(self, year: int) -> pd.DataFrame:
        url = self._URL.format(year=year)
        request = Request(url, headers={"User-Agent": "MetalPredictor research/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        frame = pd.read_csv(BytesIO(payload))
        date_col = self._find_column(frame, ("Date", "DATE"))
        two_col = self._find_column(frame, ("2 Yr", "2 YR", "2-year", "2 Year"))
        ten_col = self._find_column(frame, ("10 Yr", "10 YR", "10-year", "10 Year"))
        out = pd.DataFrame({
            "observation_date": pd.to_datetime(frame[date_col], errors="coerce").dt.normalize(),
            "rate_2y_percent": frame[two_col],
            "rate_10y_percent": frame[ten_col],
        })
        out = out.dropna(subset=["observation_date"]).reset_index(drop=True)
        return out

    @staticmethod
    def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
        normalized = {str(column).strip().lower(): column for column in frame.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in normalized:
                return normalized[key]
        raise ValueError(f"Treasury CSV missing expected column among {candidates}; got {list(frame.columns)}")
