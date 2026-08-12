from __future__ import annotations

from datetime import datetime, timezone
import math

import pytest

from metal_predictor.live.bullionvault_quote import BullionVaultQuoteProvider
from metal_predictor.live.contracts import ForecastSnapshot
from metal_predictor.live.quote_contracts import BidAskMarketQuote, MarketDepthLevel
from metal_predictor.live.spread_profit import (
    MarketQuoteProfitAssumptions,
    SpreadProfitCalculator,
)


MARKET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<envelope>
  <message type="MARKET_DEPTH_A" version="0.1">
    <market><pitches>
      <pitch securityId="AGXLN" considerationCurrency="USD">
        <buyPrices>
          <price actionIndicator="B" quantity="2.500" limit="2120"/>
          <price actionIndicator="B" quantity="4.000" limit="2119"/>
        </buyPrices>
        <sellPrices>
          <price actionIndicator="S" quantity="1.750" limit="2122"/>
          <price actionIndicator="S" quantity="3.000" limit="2123"/>
        </sellPrices>
      </pitch>
    </pitches></market>
  </message>
</envelope>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, market_xml: str = MARKET_XML) -> None:
        self.market_xml = market_xml
        self.get_calls: list[tuple[str, dict[str, object] | None]] = []
        self.post_calls: list[tuple[str, dict[str, object] | None]] = []

    def get(self, url: str, params: dict[str, object] | None = None) -> FakeResponse:
        self.get_calls.append((url, params))
        if "view_market_xml.do" in url:
            return FakeResponse(self.market_xml)
        return FakeResponse("<html><body>login</body></html>")

    def post(self, url: str, data: dict[str, object] | None = None) -> FakeResponse:
        self.post_calls.append((url, data))
        return FakeResponse("<html><body>ok</body></html>")


def _forecast() -> ForecastSnapshot:
    return ForecastSnapshot(
        feature_timestamp_utc=datetime(2026, 8, 12, 6, tzinfo=timezone.utc),
        decision_time_utc=datetime(2026, 8, 12, 7, tzinfo=timezone.utc),
        current_price_usd_per_kg=2123.40,
        baseline_model="ridge_alpha_100",
        baseline_log_return_1h=0.01,
        baseline_predicted_price_usd_per_kg=2123.40 * math.exp(0.01),
        baseline_direction="UP",
        challenger_model="ridge_alpha_10",
        challenger_log_return_1h=-0.005,
        challenger_predicted_price_usd_per_kg=2123.40 * math.exp(-0.005),
        challenger_direction="DOWN",
        data_quality="PROVIDER_AGGREGATED_H1",
        source_provider="GoldAPI",
        source_compatible_with_training=False,
        materialized_at_utc=datetime(2026, 8, 12, 7, 5, tzinfo=timezone.utc),
    )


def test_public_quote_parses_best_bid_ask_and_depth() -> None:
    client = FakeClient()
    provider = BullionVaultQuoteProvider(client=client, access_mode="public", market_width=5)

    quote = provider.fetch_quote()

    assert quote.source_provider == "BullionVault"
    assert quote.security_id == "AGXLN"
    assert quote.best_bid_usd_per_kg == 2120.0
    assert quote.best_ask_usd_per_kg == 2122.0
    assert quote.spread_usd_per_kg == 2.0
    assert quote.mid_usd_per_kg == 2121.0
    assert quote.best_bid_quantity_kg == 2.5
    assert quote.best_ask_quantity_kg == 1.75
    assert quote.access_mode == "PUBLIC_CACHED_READ_ONLY"
    assert quote.freshness_status == "SERVER_CACHED_LESS_CURRENT"
    assert len(quote.bid_depth) == 2
    assert len(quote.ask_depth) == 2
    assert client.post_calls == []
    assert "view_market_xml.do" in client.get_calls[-1][0]
    assert client.get_calls[-1][1] == {
        "considerationCurrency": "USD",
        "securityId": "AGXLN",
        "quantity": 0.001,
        "marketWidth": 5,
    }


def test_authenticated_quote_uses_login_session_but_exposes_no_order_method() -> None:
    client = FakeClient()
    provider = BullionVaultQuoteProvider(
        username="research-user",
        password="secret-value",
        access_mode="authenticated",
        client=client,
    )

    quote = provider.fetch_quote()

    assert quote.access_mode == "AUTHENTICATED_READ_ONLY"
    assert quote.freshness_status == "CURRENT_GUI_SOURCE"
    assert len(client.post_calls) == 1
    login_url, payload = client.post_calls[0]
    assert login_url.endswith("/secure/j_security_check")
    assert payload == {"j_username": "research-user", "j_password": "secret-value"}
    assert client.get_calls[-1][0].endswith("/secure/api/v2/view_market_xml.do")
    assert not hasattr(provider, "place_order")
    assert not hasattr(provider, "cancel_order")


def test_provider_rejects_xml_entity_declarations() -> None:
    client = FakeClient('<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><envelope/>')
    provider = BullionVaultQuoteProvider(client=client, access_mode="public")

    with pytest.raises(RuntimeError, match="forbidden declarations"):
        provider.fetch_quote()


def test_market_quote_projection_applies_frozen_return_to_bullionvault_mid() -> None:
    quote = BidAskMarketQuote(
        source_provider="BullionVault",
        security_id="AGXLN",
        currency="USD",
        best_bid_usd_per_kg=2120.0,
        best_ask_usd_per_kg=2122.0,
        best_bid_quantity_kg=2.5,
        best_ask_quantity_kg=1.75,
        bid_depth=(MarketDepthLevel(2120.0, 2.5),),
        ask_depth=(MarketDepthLevel(2122.0, 1.75),),
        fetched_at_utc=datetime(2026, 8, 12, 9, 10, tzinfo=timezone.utc),
        access_mode="AUTHENTICATED_READ_ONLY",
        freshness_status="CURRENT_GUI_SOURCE",
    )
    result = SpreadProfitCalculator().evaluate_market_quote(
        _forecast(),
        quote,
        MarketQuoteProfitAssumptions(target_profit_usd=10.0),
    )

    expected_baseline_mid = 2121.0 * math.exp(0.01)
    assert result["reference"]["current_bid_usd_per_kg"] == 2120.0
    assert result["reference"]["current_ask_usd_per_kg"] == 2122.0
    assert result["assumptions"]["current_spread_usd_per_kg"] == 2.0
    assert result["assumptions"]["forecast_spread_usd_per_kg"] == 2.0
    assert result["projection"]["method"] == "APPLY_FROZEN_LOG_RETURN_TO_BULLIONVAULT_MID"
    assert result["baseline"]["predicted_mid_usd_per_kg"] == pytest.approx(expected_baseline_mid)
    assert result["baseline"]["long"]["expected_margin_usd_per_kg"] == pytest.approx(
        expected_baseline_mid - 1.0 - 2122.0
    )
    assert result["baseline"]["long"]["minimum_quantity_for_target_profit_kg"] is not None
    assert result["challenger"]["short"]["positive_margin_after_assumed_costs"] is True
    assert result["safety"]["buy_sell_enabled"] is False
    assert result["safety"]["execution_enabled"] is False
    assert result["safety"]["order_submission_available"] is False
