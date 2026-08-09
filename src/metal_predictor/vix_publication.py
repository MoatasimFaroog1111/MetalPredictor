from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class VixDailyClosePublicationPolicy:
    """First safe availability of the official Cboe daily VIX closing value.

    Cboe spot VIX calculation runs through 4:15 p.m. Eastern Time. The research
    adapter therefore exposes a trading day's daily close no earlier than 16:15
    America/New_York on that same date. Hourly Silver rows consume it only through
    the completed-bar DecisionClock.
    """

    release_hour: int = 16
    release_minute: int = 15

    def available_from_utc(self, observation_dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(observation_dates, errors="raise").dt.date
        eastern = ZoneInfo("America/New_York")
        values = [
            pd.Timestamp(
                datetime.combine(
                    day,
                    time(self.release_hour, self.release_minute),
                    tzinfo=eastern,
                )
            ).tz_convert("UTC")
            for day in dates
        ]
        return pd.Series(
            pd.DatetimeIndex(values),
            index=observation_dates.index,
            dtype="datetime64[ns, UTC]",
        )
