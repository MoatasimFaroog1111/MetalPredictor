"""Research-only platinum/palladium cross-asset components."""

from metal_predictor.precious_metals.contracts import (
    HistoricalMetalSource,
    PreciousMetalInstrument,
)
from metal_predictor.precious_metals.dukascopy_source import DukascopyHistoricalMetalSource
from metal_predictor.precious_metals.features import PlatinumPalladiumCrossAssetFeatures
from metal_predictor.precious_metals.provenance import (
    HistoricalBootstrapAssessment,
    HistoricalBootstrapManifest,
    HistoricalBootstrapPolicy,
    HistoricalBootstrapProvenanceGate,
)

__all__ = [
    "HistoricalMetalSource",
    "PreciousMetalInstrument",
    "DukascopyHistoricalMetalSource",
    "PlatinumPalladiumCrossAssetFeatures",
    "HistoricalBootstrapAssessment",
    "HistoricalBootstrapManifest",
    "HistoricalBootstrapPolicy",
    "HistoricalBootstrapProvenanceGate",
]
