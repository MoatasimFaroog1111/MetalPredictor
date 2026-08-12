from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from typing import Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from metal_predictor.precious_metals.contracts import (
    JsonTransport,
    PreciousMetalInstrument,
)
from metal_predictor.price_normalization import TROY_OZ_PER_KG


class UrllibJsonTransport:
    """Small dependency-free JSON transport for Dukascopy Trading Tools API."""

    _BASE_URL = "https://freeserv.dukascopy.com/2.0/"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._timeout = float(timeout_seconds)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")

    def get_json(self, params: Mapping[str, object]) -> object:
        url = f"{self._BASE_URL}?{urlencode({k: str(v) for k, v in params.items()})}"
        request = Request(url, headers={"User-Agent": "MetalPredictor-Research/1.0"})
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except OSError:
            raise RuntimeError("Dukascopy research transport request failed.") from None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("Dukascopy research API returned invalid JSON.") from None


class DukascopyHistoricalMetalSource:
    """Research-only XPT/XPD hourly bid-OHLC source.

    The adapter uses Dukascopy's documented Trading Tools endpoints, requests UTC
    1-hour candles on the Bid side, preserves missing hours as missing, and converts
    USD/troy-ounce quotes to USD/kg. It has no trading/account methods.
    """

    _MAX_API_COUNT = 5000
    _CHUNK_HOURS = 24 * 180

    def __init__(self, api_key: str, transport: JsonTransport | None = None) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("Dukascopy API key is required for historical research data.")
        self._api_key = key
        self._transport = transport or UrllibJsonTransport()
        self._instrument_ids: dict[str, int] = {}

    def fetch_hourly(
        self,
        instrument: PreciousMetalInstrument,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        start = self._exact_hour(start_utc, "start_utc")
        end = self._exact_hour(end_utc, "end_utc")
        if end < start:
            raise ValueError("end_utc must be on or after start_utc.")

        instrument_id = self._resolve_instrument_id(instrument)
        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(hours=self._CHUNK_HOURS - 1))
            payload = self._transport.get_json({
                "path": "api/historicalPrices",
                "key": self._api_key,
                "instrument": instrument_id,
                "timeFrame": "1hour",
                "count": self._MAX_API_COUNT,
                "start": int(cursor.timestamp() * 1000),
                "end": int((chunk_end + timedelta(hours=1) - timedelta(milliseconds=1)).timestamp() * 1000),
                "dayStartTime": "UTC",
                "offerSide": "B",
            })
            frames.append(self._parse_historical_payload(payload, instrument, cursor, chunk_end))
            cursor = chunk_end + timedelta(hours=1)

        if not frames:
            return self._empty_frame()
        result = pd.concat(frames, ignore_index=True)
        if result.empty:
            return result
        result = result.sort_values("timestamp_utc").reset_index(drop=True)
        self._validate_no_conflicting_duplicates(result)
        result = result.drop_duplicates(subset=["timestamp_utc"], keep="first").reset_index(drop=True)
        return result

    def _resolve_instrument_id(self, instrument: PreciousMetalInstrument) -> int:
        cached = self._instrument_ids.get(instrument.dukascopy_name)
        if cached is not None:
            return cached
        payload = self._transport.get_json({
            "path": "api/instrumentList",
            "key": self._api_key,
            "fields": "id,name,pipValue,nameLong",
        })
        rows = self._extract_rows(payload, keys=("instruments", "data", "items"))
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name", "")).strip().upper()
            if name != instrument.dukascopy_name.upper():
                continue
            try:
                instrument_id = int(row["id"])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise RuntimeError(
                    f"Dukascopy instrument {instrument.dukascopy_name} has an invalid id."
                ) from None
            self._instrument_ids[instrument.dukascopy_name] = instrument_id
            return instrument_id
        raise RuntimeError(
            f"Dukascopy instrument list did not contain {instrument.dukascopy_name}."
        )

    def _parse_historical_payload(
        self,
        payload: object,
        instrument: PreciousMetalInstrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        rows = self._extract_rows(payload, keys=("data", "prices", "candles", "items"))
        parsed: list[dict[str, object]] = []
        for row in rows:
            candle = self._parse_candle(row)
            if candle is None:
                continue
            ts, open_oz, high_oz, low_oz, close_oz = candle
            if ts < start or ts > end:
                continue
            parsed.append({
                "timestamp_utc": ts,
                "open_usd_per_kg": open_oz * TROY_OZ_PER_KG,
                "high_usd_per_kg": high_oz * TROY_OZ_PER_KG,
                "low_usd_per_kg": low_oz * TROY_OZ_PER_KG,
                "close_usd_per_kg": close_oz * TROY_OZ_PER_KG,
                "open_usd_per_oz": open_oz,
                "high_usd_per_oz": high_oz,
                "low_usd_per_oz": low_oz,
                "close_usd_per_oz": close_oz,
                "quality_flag": "PROVIDER_H1_BID",
                "source_provider": "Dukascopy",
                "source_symbol": instrument.dukascopy_name,
                "market_type": "commodity_cfd_cross_feed",
            })
        if not parsed:
            return self._empty_frame()
        return pd.DataFrame(parsed, columns=self._columns())

    @classmethod
    def _parse_candle(cls, row: object) -> tuple[datetime, float, float, float, float] | None:
        if not isinstance(row, Mapping):
            return None
        try:
            timestamp_raw = cls._first(row, "timestamp", "time", "date", "start", "startTime")
            open_raw = cls._first(row, "open", "openPrice", "o")
            high_raw = cls._first(row, "high", "highPrice", "h")
            low_raw = cls._first(row, "low", "lowPrice", "l")
            close_raw = cls._first(row, "close", "closePrice", "c")
            ts = cls._parse_timestamp(timestamp_raw)
            prices = tuple(float(value) for value in (open_raw, high_raw, low_raw, close_raw))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if ts.minute or ts.second or ts.microsecond:
            return None
        if not all(math.isfinite(value) and value > 0 for value in prices):
            return None
        open_oz, high_oz, low_oz, close_oz = prices
        if high_oz < low_oz or high_oz < max(open_oz, close_oz) or low_oz > min(open_oz, close_oz):
            return None
        return ts, open_oz, high_oz, low_oz, close_oz

    @staticmethod
    def _extract_rows(payload: object, keys: tuple[str, ...]) -> list[object]:
        if isinstance(payload, list):
            return list(payload)
        if isinstance(payload, Mapping):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return list(value)
        raise RuntimeError("Dukascopy research API returned an unsupported response shape.")

    @staticmethod
    def _first(row: Mapping[str, object], *keys: str) -> object:
        for key in keys:
            if key in row:
                return row[key]
        raise KeyError(keys[0])

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("non-finite timestamp")
            seconds = numeric / 1000.0 if abs(numeric) >= 100_000_000_000 else numeric
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError("Dukascopy candle timestamps must include a timezone.")
        return timestamp.tz_convert("UTC").to_pydatetime()

    @staticmethod
    def _validate_no_conflicting_duplicates(frame: pd.DataFrame) -> None:
        duplicated = frame[frame["timestamp_utc"].duplicated(keep=False)]
        if duplicated.empty:
            return
        price_columns = [
            "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"
        ]
        for _, group in duplicated.groupby("timestamp_utc", sort=False):
            if len(group[price_columns].drop_duplicates()) > 1:
                raise RuntimeError("Dukascopy returned conflicting duplicate hourly candles.")

    @staticmethod
    def _exact_hour(value: datetime, name: str) -> datetime:
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware.")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError(f"{name} must align to an exact UTC hour.")
        return utc

    @staticmethod
    def _columns() -> list[str]:
        return [
            "timestamp_utc",
            "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg",
            "open_usd_per_oz", "high_usd_per_oz", "low_usd_per_oz", "close_usd_per_oz",
            "quality_flag", "source_provider", "source_symbol", "market_type",
        ]

    @classmethod
    def _empty_frame(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._columns())
