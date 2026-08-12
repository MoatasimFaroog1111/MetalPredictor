from __future__ import annotations

from metal_predictor.live.bullionvault_quote import BullionVaultQuoteProvider
from metal_predictor.live.settings import LiveSettings


def build_bullionvault_quote_provider(settings: LiveSettings) -> BullionVaultQuoteProvider:
    """Compose the read-only BullionVault adapter from deployment settings."""

    return BullionVaultQuoteProvider(
        username=settings.bullionvault_username,
        password=settings.bullionvault_password,
        security_id=settings.bullionvault_security_id,
        currency=settings.bullionvault_currency,
        minimum_quantity_kg=settings.bullionvault_minimum_quantity_kg,
        market_width=settings.bullionvault_market_width,
        access_mode=settings.bullionvault_access_mode,
        allow_public_fallback=settings.bullionvault_public_fallback,
    )
