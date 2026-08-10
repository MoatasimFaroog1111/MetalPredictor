from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


@dataclass(frozen=True)
class CftcCotPublicationPolicy:
    """Point-in-time publication policy for CFTC COT reports.

    The CFTC states that COT reports are generally released Friday at 3:30 p.m.
    Eastern using the preceding Tuesday's positions, while holidays and exceptional
    operational incidents can delay publication. Historical exact release dates are
    not comprehensively published, so this policy uses:

    1) exact official overrides for documented disruptions in the research window;
    2) a conservative holiday-week rule for otherwise normal historical weeks;
    3) Friday 15:30 ET for ordinary weeks.

    It never makes a report visible before the earliest defensible publication time.
    """

    release_hour: int = 15
    release_minute: int = 30

    _EXACT_OVERRIDES = {
        # 2021 Juneteenth observance.
        date(2021, 6, 15): date(2021, 6, 21),
        # 2023 ION cyber incident catch-up publications.
        date(2023, 1, 31): date(2023, 2, 24),
        date(2023, 2, 7): date(2023, 3, 3),
        date(2023, 2, 14): date(2023, 3, 8),
        date(2023, 2, 21): date(2023, 3, 10),
        date(2023, 2, 28): date(2023, 3, 14),
        date(2023, 3, 7): date(2023, 3, 16),
        date(2023, 3, 14): date(2023, 3, 21),
        # 2025 National Day of Mourning.
        date(2025, 1, 7): date(2025, 1, 13),
        # 2025 appropriations lapse / final accelerated catch-up schedule.
        date(2025, 9, 30): date(2025, 11, 19),
        date(2025, 10, 7): date(2025, 11, 21),
        date(2025, 10, 14): date(2025, 11, 25),
        date(2025, 10, 21): date(2025, 12, 2),
        date(2025, 10, 28): date(2025, 12, 5),
        date(2025, 11, 4): date(2025, 12, 9),
        date(2025, 11, 10): date(2025, 12, 10),
        date(2025, 11, 18): date(2025, 12, 12),
        date(2025, 11, 25): date(2025, 12, 15),
        date(2025, 12, 2): date(2025, 12, 17),
        date(2025, 12, 9): date(2025, 12, 19),
        date(2025, 12, 16): date(2025, 12, 23),
        date(2025, 12, 23): date(2025, 12, 29),
        # 2026 published delayed dates visible in the current official schedule.
        date(2025, 12, 30): date(2026, 1, 5),
        date(2026, 6, 16): date(2026, 6, 22),
        date(2026, 6, 30): date(2026, 7, 6),
    }

    @property
    def exact_override_report_dates(self) -> frozenset[date]:
        return frozenset(self._EXACT_OVERRIDES)

    def available_from_utc(self, report_dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(report_dates, errors="raise").dt.date
        values = [self._available_one(day) for day in dates]
        return pd.Series(
            pd.DatetimeIndex(values),
            index=report_dates.index,
            dtype="datetime64[ns, UTC]",
        )

    def _available_one(self, report_date: date) -> pd.Timestamp:
        release_date = self._EXACT_OVERRIDES.get(report_date)
        if release_date is None:
            friday = report_date + timedelta(days=(4 - report_date.weekday()) % 7)
            release_date = self._conservative_holiday_release(report_date, friday)
        local = datetime.combine(
            release_date,
            time(self.release_hour, self.release_minute),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return pd.Timestamp(local).tz_convert("UTC")

    def _conservative_holiday_release(self, report_date: date, friday: date) -> date:
        """Use Monday after the nominal Friday if that processing week contains a federal holiday.

        The CFTC notes that holidays may delay COT publication by one or two days and
        does not publish a complete historical release-date list. Delaying the state
        to the next Federal business day after Friday prevents look-ahead at the cost
        of conservative latency in affected weeks.
        """
        holidays = USFederalHolidayCalendar().holidays(
            start=pd.Timestamp(report_date) - pd.Timedelta(days=1),
            end=pd.Timestamp(friday),
        )
        if len(holidays) == 0:
            return friday
        business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        return (pd.Timestamp(friday) + business_day).date()
