from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class PreciousMetalInstrument:
    asset: str
    dukascopy_name: str
    output_stem: str

    def __post_init__(self) -> None:
        if self.asset.strip().upper() not in {"XPT", "XPD"}:
            raise ValueError("Precious-metal research instrument must be XPT or XPD.")
        if not self.dukascopy_name.strip() or not self.output_stem.strip():
            raise ValueError("Instrument names must be non-empty.")


PLATINUM = PreciousMetalInstrument("XPT", "XPT.CMD/USD", "XPTUSD_H1_USD_PER_KG")
PALLADIUM = PreciousMetalInstrument("XPD", "XPD.CMD/USD", "XPDUSD_H1_USD_PER_KG")


class JsonTransport(Protocol):
    def get_json(self, params: Mapping[str, object]) -> object: ...


class HistoricalMetalSource(Protocol):
    def fetch_hourly(
        self,
        instrument: PreciousMetalInstrument,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame: ...
