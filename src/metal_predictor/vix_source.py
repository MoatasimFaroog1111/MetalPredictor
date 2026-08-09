from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from metal_predictor.market_source import DownloadWindow
from metal_predictor.vix_publication import VixDailyClosePublicationPolicy


@dataclass(frozen=True)
class VixDailyReport:
    rows: int
    first_observation_date: str
    last_observation_date: str
    invalid_rows_removed: int
    source: str
    source_url: str
    current_vintage_warning: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CboeVixDailyHistoryClient:
    """Downloads official Cboe VIX daily OHLC history and attaches safe availability times."""

    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def __init__(self, publication_policy: VixDailyClosePublicationPolicy | None = None) -> None:
        self._publication = publication_policy or VixDailyClosePublicationPolicy()

    def fetch(self, window: DownloadWindow) -> tuple[pd.DataFrame, VixDailyReport]:
        request = Request(self.URL, headers={"User-Agent": "MetalPredictor research/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        raw = pd.read_csv(BytesIO(payload))
        columns = {str(column).strip().upper(): column for column in raw.columns}
        expected = ("DATE", "OPEN", "HIGH", "LOW", "CLOSE")
        missing = [name for name in expected if name not in columns]
        if missing:
            raise ValueError(f"Cboe VIX CSV missing columns {missing}; got {list(raw.columns)}")

        frame = pd.DataFrame({
            "observation_date": pd.to_datetime(raw[columns["DATE"]], errors="coerce").dt.normalize(),
            "vix_open": pd.to_numeric(raw[columns["OPEN"]], errors="coerce"),
            "vix_high": pd.to_numeric(raw[columns["HIGH"]], errors="coerce"),
            "vix_low": pd.to_numeric(raw[columns["LOW"]], errors="coerce"),
            "vix_close": pd.to_numeric(raw[columns["CLOSE"]], errors="coerce"),
        })
        start = pd.Timestamp(window.start_utc).tz_convert("UTC").tz_localize(None).normalize()
        end = pd.Timestamp(window.end_utc).tz_convert("UTC").tz_localize(None).normalize()
        frame = frame.loc[frame["observation_date"].between(start, end)].copy()
        frame = frame.sort_values("observation_date").drop_duplicates("observation_date", keep="last")
        if frame.empty:
            raise ValueError("Cboe VIX history contains no rows in requested window.")

        price_cols = ["vix_open", "vix_high", "vix_low", "vix_close"]
        values = frame[price_cols].to_numpy(float)
        valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        valid &= (frame["vix_high"] >= frame["vix_low"]).to_numpy()
        valid &= (frame["vix_high"] >= frame[["vix_open", "vix_close"]].max(axis=1)).to_numpy()
        valid &= (frame["vix_low"] <= frame[["vix_open", "vix_close"]].min(axis=1)).to_numpy()
        invalid_rows = int((~valid).sum())
        frame = frame.loc[valid].reset_index(drop=True)
        if frame.empty:
            raise ValueError("All Cboe VIX rows failed OHLC validation.")

        frame["available_from_utc"] = self._publication.available_from_utc(frame["observation_date"])
        if frame["available_from_utc"].duplicated().any():
            raise ValueError("Cboe VIX daily availability timestamps must be unique.")
        if not frame["available_from_utc"].is_monotonic_increasing:
            raise ValueError("Cboe VIX daily availability timestamps are not chronological.")

        report = VixDailyReport(
            rows=int(len(frame)),
            first_observation_date=str(frame["observation_date"].iloc[0].date()),
            last_observation_date=str(frame["observation_date"].iloc[-1].date()),
            invalid_rows_removed=invalid_rows,
            source="Cboe VIX Index Historical Data",
            source_url=self.URL,
            current_vintage_warning=(
                "Cboe publishes a current historical daily file. Daily-close availability is "
                "modeled conservatively at 16:15 America/New_York, but the file is not a "
                "historical-vintage archive of later corrections."
            ),
        )
        return frame, report
