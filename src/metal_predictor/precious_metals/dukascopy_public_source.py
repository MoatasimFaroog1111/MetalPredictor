from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from metal_predictor.precious_metals.contracts import PreciousMetalInstrument
from metal_predictor.price_normalization import TROY_OZ_PER_KG


class PublicFeedTransport(Protocol):
    """Minimal URL transport for the keyless Dukascopy historical feed."""

    def get_json(self, url: str) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class DukascopyPublicFeedSpec:
    instrument_name: str
    feed_code: str
    earliest_h1_utc: datetime

    def __post_init__(self) -> None:
        if not self.instrument_name.strip() or not self.feed_code.strip():
            raise ValueError("Dukascopy feed identifiers must be non-empty.")
        if self.earliest_h1_utc.tzinfo is None:
            raise ValueError("earliest_h1_utc must be timezone-aware.")


@dataclass(frozen=True)
class DukascopyPublicFeedBucket:
    url: str
    first_hour_utc: datetime
    last_hour_utc: datetime
    active_bucket: bool


_FEED_SPECS = {
    "XPT.CMD/USD": DukascopyPublicFeedSpec(
        instrument_name="XPT.CMD/USD",
        feed_code="XPT.CMD-USD",
        earliest_h1_utc=datetime(2021, 11, 1, 0, 0, tzinfo=timezone.utc),
    ),
    "XPD.CMD/USD": DukascopyPublicFeedSpec(
        instrument_name="XPD.CMD/USD",
        feed_code="XPD.CMD-USD",
        earliest_h1_utc=datetime(2021, 7, 4, 22, 0, tzinfo=timezone.utc),
    ),
}


class UrllibPublicFeedTransport:
    """Dependency-free read-only transport for public Dukascopy JSON history."""

    def __init__(self, timeout_seconds: float = 30.0, max_response_bytes: int = 5_000_000) -> None:
        self._timeout = float(timeout_seconds)
        self._max_response_bytes = int(max_response_bytes)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")
        if self._max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive.")

    def get_json(self, url: str) -> Mapping[str, object]:
        if not url.startswith(DukascopyPublicH1UrlPlanner.DATA_API_ROOT + "/"):
            raise ValueError("Dukascopy public-feed URL must use the fixed data API root.")
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "MetalPredictor-Research/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise RuntimeError(f"Dukascopy public feed returned HTTP {exc.code}.") from None
        except URLError:
            raise RuntimeError("Dukascopy public feed request failed.") from None
        if len(raw) > self._max_response_bytes:
            raise RuntimeError("Dukascopy public feed response exceeded the safety limit.")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("Dukascopy public feed returned invalid JSON.") from None
        if not isinstance(payload, Mapping):
            raise RuntimeError("Dukascopy public feed returned a non-object payload.")
        return payload


class DukascopyPublicH1UrlPlanner:
    """Plans month-bucket H1 URLs without model or parsing responsibilities."""

    DATA_API_ROOT = "https://jetta.dukascopy.com/v1"

    @classmethod
    def feed_spec(cls, instrument: PreciousMetalInstrument) -> DukascopyPublicFeedSpec:
        try:
            return _FEED_SPECS[instrument.dukascopy_name.upper()]
        except KeyError:
            raise ValueError(
                f"No keyless Dukascopy feed spec registered for {instrument.dukascopy_name}."
            ) from None

    def plan(
        self,
        instrument: PreciousMetalInstrument,
        start_utc: datetime,
        end_utc: datetime,
        *,
        now_utc: datetime | None = None,
    ) -> tuple[DukascopyPublicFeedBucket, ...]:
        start = self._exact_hour(start_utc, "start_utc")
        end = self._exact_hour(end_utc, "end_utc")
        if end < start:
            raise ValueError("end_utc must be on or after start_utc.")
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        spec = self.feed_spec(instrument)
        effective_start = max(start, spec.earliest_h1_utc)
        if effective_start > end:
            return ()

        cursor = datetime(effective_start.year, effective_start.month, 1, tzinfo=timezone.utc)
        buckets: list[DukascopyPublicFeedBucket] = []
        while cursor <= end:
            next_month = self._next_month(cursor)
            is_active = cursor <= now < next_month
            if is_active:
                from_ms = int(cursor.timestamp() * 1000)
                url = (
                    f"{self.DATA_API_ROOT}/candles/hour/{spec.feed_code}/BID"
                    f"?from={from_ms}"
                )
            else:
                url = (
                    f"{self.DATA_API_ROOT}/candles/hour/{spec.feed_code}/BID/"
                    f"{cursor.year}/{cursor.month}"
                )
            bucket_first = max(effective_start, cursor)
            bucket_last = min(end, next_month - pd.Timedelta(hours=1).to_pytimedelta())
            buckets.append(
                DukascopyPublicFeedBucket(
                    url=url,
                    first_hour_utc=bucket_first,
                    last_hour_utc=bucket_last,
                    active_bucket=is_active,
                )
            )
            cursor = next_month
        return tuple(buckets)

    @staticmethod
    def _next_month(value: datetime) -> datetime:
        if value.month == 12:
            return datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
        return datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _exact_hour(value: datetime, name: str) -> datetime:
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware.")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError(f"{name} must align to an exact UTC hour.")
        return utc


class DukascopyCompressedH1Decoder:
    """Pure decoder for Dukascopy's delta-compressed JSON candle columns.

    Missing source hours stay missing. Unlike generic display-oriented clients, this
    decoder never synthesizes flat candles to fill time gaps.
    """

    H1_SHIFT_MS = 3_600_000

    def decode(self, payload: Mapping[str, object]) -> tuple[tuple[datetime, float, float, float, float], ...]:
        times = self._numeric_array(payload, "times", integer=True, non_negative=True)
        if not times:
            return ()
        multiplier = self._positive_float(payload.get("multiplier"), "multiplier")
        timestamp_ms = self._non_negative_integer(payload.get("timestamp"), "timestamp")
        shift_ms = self._positive_integer(payload.get("shift"), "shift")
        if shift_ms != self.H1_SHIFT_MS:
            raise RuntimeError(
                f"Dukascopy public feed returned shift={shift_ms}; expected exact H1 shift."
            )

        base = {
            name: self._positive_float(payload.get(name), name)
            for name in ("open", "high", "low", "close")
        }
        deltas = {
            name: self._numeric_array(payload, plural, integer=True)
            for name, plural in (
                ("open", "opens"),
                ("high", "highs"),
                ("low", "lows"),
                ("close", "closes"),
            )
        }
        expected = len(times)
        if any(len(values) != expected for values in deltas.values()):
            raise RuntimeError("Dukascopy public feed candle columns have mismatched lengths.")

        units = {name: int(round(value / multiplier)) for name, value in base.items()}
        rows: list[tuple[datetime, float, float, float, float]] = []
        current_ms = timestamp_ms
        for index, time_delta in enumerate(times):
            current_ms += int(time_delta) * shift_ms
            for name in units:
                units[name] += int(deltas[name][index])
            prices = tuple(units[name] * multiplier for name in ("open", "high", "low", "close"))
            if not all(math.isfinite(value) and value > 0 for value in prices):
                raise RuntimeError("Dukascopy public feed produced an invalid price.")
            open_oz, high_oz, low_oz, close_oz = prices
            if high_oz < low_oz or high_oz < max(open_oz, close_oz) or low_oz > min(open_oz, close_oz):
                raise RuntimeError("Dukascopy public feed produced an invalid OHLC candle.")
            timestamp = datetime.fromtimestamp(current_ms / 1000.0, tz=timezone.utc)
            if timestamp.minute or timestamp.second or timestamp.microsecond:
                raise RuntimeError("Dukascopy public feed produced a non-exact UTC H1 timestamp.")
            rows.append((timestamp, open_oz, high_oz, low_oz, close_oz))
        return tuple(rows)

    @staticmethod
    def _positive_float(value: object, name: str) -> float:
        if isinstance(value, bool):
            raise RuntimeError(f"Dukascopy public feed field {name} is invalid.")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError(f"Dukascopy public feed field {name} is invalid.") from None
        if not math.isfinite(result) or result <= 0:
            raise RuntimeError(f"Dukascopy public feed field {name} must be positive.")
        return result

    @classmethod
    def _positive_integer(cls, value: object, name: str) -> int:
        result = cls._integer(value, name)
        if result <= 0:
            raise RuntimeError(f"Dukascopy public feed field {name} must be positive.")
        return result

    @classmethod
    def _non_negative_integer(cls, value: object, name: str) -> int:
        result = cls._integer(value, name)
        if result < 0:
            raise RuntimeError(f"Dukascopy public feed field {name} must be non-negative.")
        return result

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Dukascopy public feed field {name} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise RuntimeError(f"Dukascopy public feed field {name} must be an integer.")
        return int(numeric)

    @classmethod
    def _numeric_array(
        cls,
        payload: Mapping[str, object],
        name: str,
        *,
        integer: bool,
        non_negative: bool = False,
    ) -> tuple[float, ...]:
        raw = payload.get(name)
        if not isinstance(raw, list):
            raise RuntimeError(f"Dukascopy public feed field {name} must be an array.")
        values: list[float] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"Dukascopy public feed field {name} contains invalid values.")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"Dukascopy public feed field {name} contains non-finite values.")
            if integer and not numeric.is_integer():
                raise RuntimeError(f"Dukascopy public feed field {name} must contain integers.")
            if non_negative and numeric < 0:
                raise RuntimeError(f"Dukascopy public feed field {name} must be non-negative.")
            values.append(numeric)
        return tuple(values)


class DukascopyPublicHistoricalMetalSource:
    """Keyless, read-only XPT/XPD H1 Bid source for research only."""

    def __init__(
        self,
        transport: PublicFeedTransport | None = None,
        planner: DukascopyPublicH1UrlPlanner | None = None,
        decoder: DukascopyCompressedH1Decoder | None = None,
    ) -> None:
        self._transport = transport or UrllibPublicFeedTransport()
        self._planner = planner or DukascopyPublicH1UrlPlanner()
        self._decoder = decoder or DukascopyCompressedH1Decoder()

    def fetch_hourly(
        self,
        instrument: PreciousMetalInstrument,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        start = self._planner._exact_hour(start_utc, "start_utc")
        end = self._planner._exact_hour(end_utc, "end_utc")
        if end < start:
            raise ValueError("end_utc must be on or after start_utc.")

        parsed: list[dict[str, object]] = []
        for bucket in self._planner.plan(instrument, start, end):
            payload = self._transport.get_json(bucket.url)
            for ts, open_oz, high_oz, low_oz, close_oz in self._decoder.decode(payload):
                if ts < bucket.first_hour_utc or ts > bucket.last_hour_utc:
                    continue
                parsed.append(
                    {
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
                        "source_provider": "Dukascopy Public Historical Feed",
                        "source_symbol": instrument.dukascopy_name,
                        "market_type": "commodity_cfd_cross_feed",
                    }
                )
        if not parsed:
            return self._empty_frame()
        frame = pd.DataFrame(parsed, columns=self._columns())
        frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
        self._validate_duplicates(frame)
        return frame.drop_duplicates(subset=["timestamp_utc"], keep="first").reset_index(drop=True)

    @staticmethod
    def _validate_duplicates(frame: pd.DataFrame) -> None:
        duplicated = frame[frame["timestamp_utc"].duplicated(keep=False)]
        if duplicated.empty:
            return
        price_columns = [
            "open_usd_per_kg",
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
        ]
        for _, group in duplicated.groupby("timestamp_utc", sort=False):
            if len(group[price_columns].drop_duplicates()) > 1:
                raise RuntimeError("Dukascopy public feed returned conflicting duplicate H1 candles.")

    @staticmethod
    def _columns() -> list[str]:
        return [
            "timestamp_utc",
            "open_usd_per_kg",
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
            "open_usd_per_oz",
            "high_usd_per_oz",
            "low_usd_per_oz",
            "close_usd_per_oz",
            "quality_flag",
            "source_provider",
            "source_symbol",
            "market_type",
        ]

    @classmethod
    def _empty_frame(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._columns())
