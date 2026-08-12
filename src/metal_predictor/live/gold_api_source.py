from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import httpx2 as httpx

from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.price_normalization import TROY_OZ_PER_KG


class GoldApiSilverOhlcSource:
    """Fetch completed XAG hourly OHLC from Gold API.

    Gold API is an operational cross-feed, not a replacement for the frozen HistData
    research dataset. The free tier limits OHLC/history requests, so range requests
    intentionally fetch only the newest requested completed hour. Any earlier outage
    remains an explicit time gap; the frozen feature graph already preserves exact-clock
    missing lags and the sealed model imputer handles those NaNs without fabrication.
    """

    _BASE_URL = "https://api.gold-api.com"

    def __init__(
        self,
        api_key: str,
        symbol: str = "XAG",
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("Gold API key is required.")
        normalized_symbol = symbol.strip().upper() or "XAG"
        if normalized_symbol != "XAG":
            raise ValueError("Gold API live Silver source requires symbol XAG.")
        self._api_key = key
        self._symbol = normalized_symbol
        self._client = client or httpx.Client(timeout=30.0)

    def fetch_completed_hour(self, hour_start_utc: datetime) -> HourlySilverBar:
        start = self._hour_start(hour_start_utc)
        end = start + timedelta(hours=1) - timedelta(seconds=1)
        payload = self._request_ohlc(start, end)
        return self._bar_from_payload(payload, start)

    def fetch_completed_range(
        self,
        start_hour_utc: datetime,
        end_hour_utc: datetime,
    ) -> list[HourlySilverBar]:
        start = self._hour_start(start_hour_utc)
        end = self._hour_start(end_hour_utc)
        if end < start:
            raise ValueError("end_hour_utc must be on or after start_hour_utc.")

        # Free-tier-safe operational policy: fetch the newest completed hour only.
        # We preserve any older missing hours as gaps instead of inventing prices or
        # exhausting the provider's hourly OHLC allowance during startup catch-up.
        return [self.fetch_completed_hour(end)]

    def _request_ohlc(self, start: datetime, end: datetime) -> dict[str, object]:
        try:
            response = self._client.get(
                f"{self._BASE_URL}/ohlc/{self._symbol}",
                params={
                    "startTimestamp": int(start.timestamp()),
                    "endTimestamp": int(end.timestamp()),
                },
                headers={"x-api-key": self._api_key},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Gold API request failed with HTTP {exc.response.status_code}."
            ) from None
        except httpx.HTTPError:
            raise RuntimeError("Gold API transport request failed.") from None

        try:
            payload = response.json()
        except (TypeError, ValueError):
            raise RuntimeError("Gold API returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise RuntimeError("Gold API returned a non-object response.")
        return payload

    def _bar_from_payload(
        self,
        payload: dict[str, object],
        timestamp_utc: datetime,
    ) -> HourlySilverBar:
        required = ("open", "high", "low", "close")
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"Gold API OHLC response missing fields: {missing}")

        try:
            open_oz, high_oz, low_oz, close_oz = (
                float(payload[name]) for name in required
            )
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("Gold API OHLC response contains invalid price values.") from None

        prices = (open_oz, high_oz, low_oz, close_oz)
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise RuntimeError("Gold API OHLC response contains non-positive or non-finite prices.")
        if high_oz < low_oz or high_oz < max(open_oz, close_oz) or low_oz > min(open_oz, close_oz):
            raise RuntimeError("Gold API OHLC response violates OHLC invariants.")

        return HourlySilverBar(
            timestamp_utc=timestamp_utc,
            open_usd_per_kg=open_oz * TROY_OZ_PER_KG,
            high_usd_per_kg=high_oz * TROY_OZ_PER_KG,
            low_usd_per_kg=low_oz * TROY_OZ_PER_KG,
            close_usd_per_kg=close_oz * TROY_OZ_PER_KG,
            minute_count=1,
            quality_flag="PROVIDER_AGGREGATED_H1",
            source_provider="GoldAPI",
            source_symbol=self._symbol,
            market_type="spot_quote",
        )

    @staticmethod
    def _hour_start(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("hour_start_utc must be timezone-aware.")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError("hour_start_utc must align to an exact UTC hour.")
        return utc
