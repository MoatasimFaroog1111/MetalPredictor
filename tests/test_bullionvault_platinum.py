"""Synthetic XML fixtures test parsing only; they are not real market observations."""
import pytest

from metal_predictor.live.bullionvault_quote import BullionVaultQuoteProvider


class Response:
    text = """<envelope><pitch securityId="PTXLN" considerationCurrency="USD">
    <buyPrices><price quantity="0.01" limit="50000"/></buyPrices>
    <sellPrices><price quantity="0.02" limit="50100"/></sellPrices>
    </pitch></envelope>"""

    def raise_for_status(self):
        pass


class ReadOnlyClient:
    def get(self, url, params=None):
        assert url.endswith("/view_market_xml.do")
        assert params["securityId"] == "PTXLN"
        return Response()

    def post(self, *args, **kwargs):
        raise AssertionError("No POST allowed in this public read-only test")


def test_platinum_public_quote_with_synthetic_fixture():
    provider = BullionVaultQuoteProvider(
        security_id="PTXLN", access_mode="public", client=ReadOnlyClient()
    )
    quote = provider.fetch_quote()
    assert quote.security_id == "PTXLN"
    assert quote.mid_usd_per_kg == 50050
    assert quote.freshness_status == "SERVER_CACHED_LESS_CURRENT"
    assert not hasattr(provider, "place_order")


def test_unsupported_security_still_rejected():
    with pytest.raises(ValueError):
        BullionVaultQuoteProvider(security_id="INVALID", client=ReadOnlyClient())
