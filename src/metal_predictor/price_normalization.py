from __future__ import annotations

from typing import Protocol

import pandas as pd


TROY_OZ_PER_KG = 32.15074656862798


class HourlyPriceNormalizer(Protocol):
    """Strategy contract for converting source quote units into model-ready values."""

    @property
    def value_unit(self) -> str: ...

    def normalize(self, hourly: pd.DataFrame) -> pd.DataFrame: ...


class PreciousMetalUsdKgNormalizer:
    """HistData XAU/XAG source quotes are USD/troy oz; expose both oz and kg representations."""

    @property
    def value_unit(self) -> str:
        return "USD/kg"

    def normalize(self, hourly: pd.DataFrame) -> pd.DataFrame:
        out = hourly.copy(deep=True)
        for side in ("open", "high", "low", "close"):
            source = out[f"{side}_source"]
            out[f"{side}_usd_per_oz"] = source
            out[f"{side}_usd_per_kg"] = source * TROY_OZ_PER_KG
            out[f"{side}_value"] = out[f"{side}_usd_per_kg"]
        return out


class IdentityIndexNormalizer:
    """Preserves an index feed in native index points without pretending it is a metal price."""

    @property
    def value_unit(self) -> str:
        return "index_points"

    def normalize(self, hourly: pd.DataFrame) -> pd.DataFrame:
        out = hourly.copy(deep=True)
        for side in ("open", "high", "low", "close"):
            out[f"{side}_value"] = out[f"{side}_source"]
        return out
