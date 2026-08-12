"""Research-only platinum/palladium cross-asset components."""

from metal_predictor.precious_metals.contracts import (
    HistoricalMetalSource,
    PreciousMetalInstrument,
)
from metal_predictor.precious_metals.dukascopy_source import DukascopyHistoricalMetalSource
from metal_predictor.precious_metals.features import PlatinumPalladiumCrossAssetFeatures

__all__ = [
    "HistoricalMetalSource",
    "PreciousMetalInstrument",
    "DukascopyHistoricalMetalSource",
    "PlatinumPalladiumCrossAssetFeatures",
]
