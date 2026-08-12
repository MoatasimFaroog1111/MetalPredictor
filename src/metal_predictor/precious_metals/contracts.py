from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class PreciousMetalInstrument:
    """Immutable research definition for one auxiliary precious-metal market."""

    asset: str
    dukascopy_name: str
    output_stem: str

    def __post_init__(self) -> None:
        asset = self.asset.strip().upper()
        if asset not in {"XPT", "XPD"}:
            raise ValueError("Precious-metal research instrument must be XPT or XPD.")
        if not self.dukascopy_name.strip() or not self.output_stem.strip():
            raise ValueError("Instrument names must be non-empty.")


PLATINUM = PreciousMetalInstrument(
    asset="XPT",
    dukascopy_name="XPT.CMD/USD",
    output_stem="XPTUSD_H1_USD_PER_KG",
)
PALLADIUM = PreciousMetalInstrument(
    asset="XPD",
    dukascopy_name="XPD.CMD/USD",
    output_stem="XPDUSD_H1_USD_PER_KG",
)


class JsonTransport(Protocol):
    """Minimal provider transport so source parsing can be tested without network I/O."""

    def get_json(self, params: Mapping[str, object]) -> object: ...


class HistoricalMetalSource(Protocol):
    """Provider-neutral contract for exact-timestamp research OHLC."""

    def fetch_hourly(
        self,
        instrument: PreciousMetalInstrument,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame: ...
