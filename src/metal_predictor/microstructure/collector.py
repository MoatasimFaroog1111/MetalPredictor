from __future__ import annotations

from metal_predictor.live.quote_contracts import MarketQuoteProvider
from metal_predictor.microstructure.contracts import (
    MicrostructureRepository,
    MicrostructureResearchRecord,
    MicrostructureSnapshot,
)
from metal_predictor.microstructure.features import MicrostructureFeatureBuilder


class MicrostructureResearchCollector:
    """Orchestrates one read-only quote capture and append-only research write."""

    def __init__(
        self,
        provider: MarketQuoteProvider,
        repository: MicrostructureRepository,
        feature_builder: MicrostructureFeatureBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._repository = repository
        self._features = feature_builder or MicrostructureFeatureBuilder()

    def collect_once(self) -> MicrostructureResearchRecord:
        quote = self._provider.fetch_quote()
        snapshot = MicrostructureSnapshot.from_quote(quote)
        previous = self._repository.latest_snapshot()
        features = self._features.build(snapshot, previous)
        return self._repository.append(snapshot, features)
