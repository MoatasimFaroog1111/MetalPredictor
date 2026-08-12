from __future__ import annotations

from metal_predictor.live.contracts import MarketBarBackfillSource
from metal_predictor.live.gold_api_source import GoldApiSilverOhlcSource
from metal_predictor.live.market_sources import TwelveDataSilverMinuteSource
from metal_predictor.live.settings import LiveSettings


def build_live_market_source(config: LiveSettings) -> MarketBarBackfillSource | None:
    """Compose the configured operational market-data adapter.

    Provider-specific construction stays outside the FastAPI delivery layer so adding
    or replacing feeds does not change inference, persistence, or route behavior.
    """

    provider = config.resolved_market_provider
    if provider == "GoldAPI":
        return GoldApiSilverOhlcSource(
            config.gold_api_key,
            config.gold_api_symbol,
        )
    if provider == "TwelveData":
        return TwelveDataSilverMinuteSource(
            config.twelvedata_api_key,
            config.twelvedata_symbol,
        )
    return None
