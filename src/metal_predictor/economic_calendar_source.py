from __future__ import annotations

from dataclasses import dataclass
import json
import os
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(frozen=True)
class EconomicCalendarAccessReport:
    rows: int
    first_release_utc: str | None
    last_release_utc: str | None
    with_actual: int
    with_consensus_forecast: int
    source: str
    authentication_mode: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "first_release_utc": self.first_release_utc,
            "last_release_utc": self.last_release_utc,
            "with_actual": self.with_actual,
            "with_consensus_forecast": self.with_consensus_forecast,
            "source": self.source,
            "authentication_mode": self.authentication_mode,
        }


class TradingEconomicsCalendarClient:
    """Read-only point-in-time economic-calendar adapter.

    Trading Economics documents the calendar Date field as UTC and Forecast as the
    survey consensus from a representative group of economists. The adapter accepts
    a paid key through TRADING_ECONOMICS_API_KEY and otherwise uses the documented
    limited `guest:guest` trial identity. It never stores credentials in repository
    files and fails closed when the account lacks historical-calendar permission.
    """

    _BASE = "https://api.tradingeconomics.com"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TRADING_ECONOMICS_API_KEY") or "guest:guest"
        self._auth_mode = "environment_api_key" if self._api_key != "guest:guest" else "guest_limited"

    def fetch_us_calendar(
        self,
        start_date: str,
        end_date: str,
        importance: int = 3,
    ) -> tuple[pd.DataFrame, EconomicCalendarAccessReport]:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("Invalid economic-calendar date range.")
        country = quote("united states")
        url = (
            f"{self._BASE}/calendar/country/{country}/"
            f"{start.date().isoformat()}/{end.date().isoformat()}"
            f"?c={quote(self._api_key, safe='')}&importance={int(importance)}&values=true&f=json"
        )
        request = Request(url, headers={"User-Agent": "MetalPredictor research/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        data = json.loads(payload.decode("utf-8"))
        if isinstance(data, dict) and any(
            token in str(data).lower() for token in ("error", "not authorized", "permission", "subscription")
        ):
            raise PermissionError(f"Trading Economics calendar access rejected: {data}")
        if not isinstance(data, list):
            raise ValueError(f"Unexpected Trading Economics response type: {type(data).__name__}")
        frame = pd.DataFrame(data)
        if frame.empty:
            raise PermissionError(
                "Trading Economics returned no historical U.S. calendar rows. "
                "The guest account may not include this historical range."
            )
        required = {"Date", "Country", "Event", "Actual", "Forecast", "Importance"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Economic calendar response missing columns: {sorted(missing)}")
        frame["release_utc"] = pd.to_datetime(frame["Date"], utc=True, errors="coerce")
        frame = frame.loc[frame["release_utc"].notna()].copy()
        frame = frame.sort_values("release_utc").reset_index(drop=True)
        report = EconomicCalendarAccessReport(
            rows=int(len(frame)),
            first_release_utc=(frame["release_utc"].min().isoformat() if len(frame) else None),
            last_release_utc=(frame["release_utc"].max().isoformat() if len(frame) else None),
            with_actual=int(frame["Actual"].astype("string").str.strip().ne("").sum()),
            with_consensus_forecast=int(frame["Forecast"].astype("string").str.strip().ne("").sum()),
            source="Trading Economics Economic Calendar Point-in-Time",
            authentication_mode=self._auth_mode,
        )
        return frame, report
