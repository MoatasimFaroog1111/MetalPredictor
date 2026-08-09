from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


@dataclass(frozen=True)
class H15PublicationPolicy:
    """Maps a Treasury observation date to the first H.15 time it was safely public.

    Normal policy: next Federal business day at 4:15 p.m. America/New_York.
    Explicit exceptions encode Federal Reserve Board closures and documented H.15
    Treasury-data publication delays during the research window.
    """

    release_hour: int = 16
    release_minute: int = 15

    _BOARD_CLOSURES = frozenset({date(2025, 1, 9)})
    _TREASURY_DELAY_OVERRIDES = {
        date(2023, 8, 1): date(2023, 8, 3),
        date(2023, 9, 12): date(2023, 9, 14),
    }

    def available_from_utc(self, observation_dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(observation_dates, errors="raise").dt.date
        results = [self._one(day) for day in dates]
        return pd.Series(pd.DatetimeIndex(results), index=observation_dates.index, dtype="datetime64[ns, UTC]")

    def _one(self, observation_date: date) -> pd.Timestamp:
        override = self._TREASURY_DELAY_OVERRIDES.get(observation_date)
        release_date = override or self._next_release_business_day(observation_date)
        local = datetime.combine(
            release_date,
            time(self.release_hour, self.release_minute),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return pd.Timestamp(local).tz_convert("UTC")

    def _next_release_business_day(self, observation_date: date) -> date:
        business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        candidate = (pd.Timestamp(observation_date) + business_day).date()
        while candidate in self._BOARD_CLOSURES:
            candidate = (pd.Timestamp(candidate) + business_day).date()
        return candidate
