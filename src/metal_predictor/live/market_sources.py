from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx2 as httpx
import numpy as np
import pandas as pd

from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import DownloadWindow, InstrumentSpec


class TwelveDataSilverMinuteSource:
    """Fetch one completed XAG/USD hour as 1-minute bars, then reuse canonical aggregation.

    Twelve Data is intentionally an operational feed adapter, not a replacement research
    dataset. The resulting source metadata remains explicit because its quote convention
    is not assumed identical to the HistData spot-bid training feed.
    """

    _URL = "https://api.twelvedata.com/time_series"

    def __init__(
        self,
        api_key: str,
        symbol: str = "XAG/USD",
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Twelve Data API key is required.")
        self._api_key = api_key.strip()
        self._symbol = symbol.strip() or "XAG/USD"
        self._client = client or httpx.Client(timeout=30.0)
        self._aggregator = ConservativeH1Aggregator()

    def fetch_completed_hour(self, hour_start_utc: datetime) -> HourlySilverBar:
        start = self._hour_start(hour_start_utc)
        end = start + timedelta(minutes=59, seconds=59)
        response = self._client.get(
            self._URL,
            params={
                "symbol": self._symbol,
                "interval": "1min",
                "start_date": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_date": end.strftime("%Y-%m-%dT%H:%M:%S"),
                "timezone": "UTC",
                "order": "asc",
                "outputsize": 60,
                "format": "JSON",
                "apikey": self._api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise RuntimeError(
                f"Twelve Data error {payload.get('code', 'unknown')}: "
                f"{payload.get('message', 'unknown error')}"
            )
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Twelve Data returned no XAG/USD minute bars for completed hour.")

        minutes = self._minutes_frame(values)
        instrument = InstrumentSpec(
            asset="XAG",
            pair="xagusd",
            source_symbol=self._symbol,
            provider="TwelveData",
            market_type="spot_quote",
        )
        window = DownloadWindow(
            start_utc=pd.Timestamp(start),
            end_utc=pd.Timestamp(end),
        )
        hourly, _ = self._aggregator.aggregate(minutes, instrument, window)
        matching = hourly.loc[
            pd.to_datetime(hourly["timestamp_utc"], utc=True).eq(pd.Timestamp(start))
        ]
        if len(matching) != 1:
            raise RuntimeError("Twelve Data aggregation did not produce exactly one requested H1 bar.")
        row = matching.iloc[0]
        return HourlySilverBar(
            timestamp_utc=start,
            open_usd_per_kg=float(row["open_usd_per_kg"]),
            high_usd_per_kg=float(row["high_usd_per_kg"]),
            low_usd_per_kg=float(row["low_usd_per_kg"]),
            close_usd_per_kg=float(row["close_usd_per_kg"]),
            minute_count=int(row["minute_count"]),
            quality_flag=str(row["quality_flag"]),
            source_provider=str(row["source_provider"]),
            source_symbol=str(row["source_symbol"]),
            market_type=str(row["market_type"]),
        )

    @staticmethod
    def _minutes_frame(values: list[dict[str, object]]) -> pd.DataFrame:
        frame = pd.DataFrame(values)
        required = {"datetime", "open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Twelve Data minute response missing fields: {sorted(missing)}")
        frame["timestamp_utc"] = pd.to_datetime(
            frame["datetime"], utc=True, errors="coerce"
        )
        if frame["timestamp_utc"].isna().any():
            raise RuntimeError("Twelve Data returned invalid minute timestamps.")
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "volume" not in frame.columns:
            frame["volume"] = np.nan
        else:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        prices = frame[["open", "high", "low", "close"]]
        finite = np.isfinite(prices.to_numpy(float)).all(axis=1)
        positive = prices.gt(0).all(axis=1)
        invariants = (
            frame["high"].ge(frame["low"])
            & frame["high"].ge(frame[["open", "close"]].max(axis=1))
            & frame["low"].le(frame[["open", "close"]].min(axis=1))
        )
        frame["minute_valid_ohlc"] = finite & positive & invariants
        frame["archive_sequence"] = 0
        frame["source_row_number"] = np.arange(len(frame), dtype=np.int64)
        return frame.sort_values("timestamp_utc").reset_index(drop=True)

    @staticmethod
    def _hour_start(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("hour_start_utc must be timezone-aware.")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError("hour_start_utc must align to an exact UTC hour.")
        return utc
