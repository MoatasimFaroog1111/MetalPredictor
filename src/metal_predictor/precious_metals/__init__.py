"""Research-only platinum/palladium cross-asset components."""

from metal_predictor.precious_metals.ablation import (
    ABLATION_VERSION,
    FamilyEvidencePolicy,
    PreciousMetalsFeatureFamily,
    PreciousMetalsFeatureFamilyRegistry,
)
from metal_predictor.precious_metals.confirmation import (
    CANDIDATE_FEATURES,
    CANDIDATE_FAMILIES,
    CANDIDATE_ID,
    CONFIRMATION_VERSION,
    HistoricalConfirmationPolicy,
    candidate_fingerprint,
)
from metal_predictor.precious_metals.contracts import (
    HistoricalMetalSource,
    PreciousMetalInstrument,
)
from metal_predictor.precious_metals.dukascopy_public_source import (
    DukascopyCompressedH1Decoder,
    DukascopyPublicH1UrlPlanner,
    DukascopyPublicHistoricalMetalSource,
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
    "ABLATION_VERSION",
    "FamilyEvidencePolicy",
    "PreciousMetalsFeatureFamily",
    "PreciousMetalsFeatureFamilyRegistry",
    "CANDIDATE_FEATURES",
    "CANDIDATE_FAMILIES",
    "CANDIDATE_ID",
    "CONFIRMATION_VERSION",
    "HistoricalConfirmationPolicy",
    "candidate_fingerprint",
    "HistoricalMetalSource",
    "PreciousMetalInstrument",
    "DukascopyHistoricalMetalSource",
    "DukascopyPublicHistoricalMetalSource",
    "DukascopyPublicH1UrlPlanner",
    "DukascopyCompressedH1Decoder",
    "PlatinumPalladiumCrossAssetFeatures",
    "HistoricalBootstrapAssessment",
    "HistoricalBootstrapManifest",
    "HistoricalBootstrapPolicy",
    "HistoricalBootstrapProvenanceGate",
]
