from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class DecisionClock(Protocol):
    """Maps a bar label to the instant at which all features in that bar are observable."""

    def available_at(self, bar_timestamps: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class CompletedHourlyBarDecisionClock:
    """The canonical XAG H1 timestamp labels the start of an hourly bar.

    OHLC features for row t are only known when that bar completes one hour later.
    Therefore auxiliary published information released during [t, t+1h] may be used
    for the forecast made after the current bar closes, while anything after t+1h is
    future information and remains forbidden.
    """

    bar_hours: int = 1

    def __post_init__(self) -> None:
        if self.bar_hours != 1:
            raise ValueError("The current canonical dataset supports exactly 1-hour bars.")

    def available_at(self, bar_timestamps: pd.Series) -> pd.Series:
        starts = pd.to_datetime(bar_timestamps, utc=True, errors="raise").astype(
            "datetime64[ns, UTC]"
        )
        return starts + pd.Timedelta(hours=self.bar_hours)
