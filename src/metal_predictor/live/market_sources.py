from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx2 as httpx
import numpy as np
import pandas as pd

from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import DownloadWindow, InstrumentSpec


class TwelveDataSilverMinuteSource:
    """Fetch XAG/USD M1 data and reuse the canonical conservative H1 aggregator.

    Twelve Data is an operational feed adapter, not a replacement research dataset.
    Its quote convention is not assumed identical to the HistData spot-bid training feed.
    Historical catch-up requests are bounded to 72 hours (<=4,320 one-minute points),
    below Twelve Data's documented 5,000-point single-request ceiling.
    """

    _URL = "https://api.twelvedata.com/time_series"
    _MAX_HOURS_PER_REQUEST = 72

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
        self._instrument = InstrumentSpec(
            asset="XAG",
            pair="xagusd",
            source_symbol=self._symbol,
            provider="TwelveData",
            market_type="spot_quote",
        )

    def fetch_completed_hour(self, hour_start_utc: datetime) -> HourlySilverBar:
        start = self._hour_start(hour_start_utc)
        bars = self.fetch_completed_range(start, start)
        matching = [item for item in bars if item.timestamp_utc == start]
        if len(matching) != 1:
            raise RuntimeError(
                "Twelve Data returned no usable XAG/USD H1 bar for the requested completed hour."
            )
        return matching[0]

    def fetch_completed_range(
        self,
        start_hour_utc: datetime,
        end_hour_utc: datetime,
    ) -> list[HourlySilverBar]:
        start = self._hour_start(start_hour_utc)
        end = self._hour_start(end_hour_utc)
        if end < start:
            raise ValueError("end_hour_utc must be on or after start_hour_utc.")

        output: list[HourlySilverBar] = []
        cursor = start
        while cursor <= end:
            chunk_last = min(
                end,
                cursor + timedelta(hours=self._MAX_HOURS_PER_REQUEST - 1),
            )
            request_end = chunk_last + timedelta(minutes=59, seconds=59)
            values = self._request_values(cursor, request_end)
            if values:
                output.extend(self._aggregate_values(values, cursor, request_end))
            cursor = chunk_last + timedelta(hours=1)

        ordered = sorted(output, key=lambda item: item.timestamp_utc)
        timestamps = [item.timestamp_utc for item in ordered]
        if len(timestamps) != len(set(timestamps)):
            raise RuntimeError("Twelve Data catch-up produced duplicate H1 timestamps.")
        if any(item.timestamp_utc < start or item.timestamp_utc > end for item in ordered):
            raise RuntimeError("Twelve Data catch-up produced an H1 bar outside the requested range.")
        return ordered

    def _request_values(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, object]]:
        try:
            response = self._client.get(
                self._URL,
                params={
                    "symbol": self._symbol,
                    "interval": "1min",
                    "start_date": start.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end_date": end.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timezone": "UTC",
                    "order": "asc",
                    "format": "JSON",
                    "apikey": self._api_key,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise RuntimeError("Twelve Data transport request failed.") from None

        try:
            payload = response.json()
        except (TypeError, ValueError):
            raise RuntimeError("Twelve Data returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise RuntimeError("Twelve Data returned a non-object response.")
        if payload.get("status") == "error":
            code = str(payload.get("code", "unknown"))
            raise RuntimeError(f"Twelve Data API rejected the request with code {code}.")
        values = payload.get("values")
        if values is None:
            return []
        if not isinstance(values, list):
            raise RuntimeError("Twelve Data response field 'values' is not a list.")
        if not all(isinstance(item, dict) for item in values):
            raise RuntimeError("Twelve Data returned malformed minute rows.")
        return values

    def _aggregate_values(
        self,
        values: list[dict[str, object]],
        start: datetime,
        end: datetime,
    ) -> list[HourlySilverBar]:
        minutes = self._minutes_frame(values)
        window = DownloadWindow(
            start_utc=pd.Timestamp(start),
            end_utc=pd.Timestamp(end),
        )
        hourly, _ = self._aggregator.aggregate(minutes, self._instrument, window)
        result: list[HourlySilverBar] = []
        for row in hourly.sort_values("timestamp_utc").itertuples(index=False):
            timestamp = pd.Timestamp(row.timestamp_utc).tz_convert("UTC").to_pydatetime()
            result.append(
                HourlySilverBar(
                    timestamp_utc=timestamp,
                    open_usd_per_kg=float(row.open_usd_per_kg),
                    high_usd_per_kg=float(row.high_usd_per_kg),
                    low_usd_per_kg=float(row.low_usd_per_kg),
                    close_usd_per_kg=float(row.close_usd_per_kg),
                    minute_count=int(row.minute_count),
                    quality_flag=str(row.quality_flag),
                    source_provider=str(row.source_provider),
                    source_symbol=str(row.source_symbol),
                    market_type=str(row.market_type),
                )
            )
        return result

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
