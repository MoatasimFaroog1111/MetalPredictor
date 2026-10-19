"""Offline PTXLN adapter tests using synthetic, non-market XML and credentials.

Adapter support does not enable platinum forecasting: LiveSettings intentionally
remains silver-only, and no tests reconfigure the silver application for platinum.
"""
from datetime import datetime, timedelta, timezone
import socket
from urllib.parse import parse_qs

import httpx2 as httpx
import pytest

from metal_predictor.live.bullionvault_quote import BullionVaultQuoteProvider
from metal_predictor.live.settings import LiveSettings
from metal_predictor.microstructure.contracts import MicrostructureSnapshot
from metal_predictor.microstructure.features import MicrostructureFeatureBuilder
from metal_predictor.microstructure.persisted_quote import PersistedMicrostructureQuoteProvider
from metal_predictor.microstructure.repository import SQLiteMicrostructureRepository


def pitch(security="PTXLN", currency="USD", bids=None, asks=None):
    if bids is None:
        bids = '<price quantity="0.01" limit="50000"/>'
    if asks is None:
        asks = '<price quantity="0.02" limit="50100"/>'
    return (
        f'<pitch securityId="{security}" considerationCurrency="{currency}">'
        f"<buyPrices>{bids}</buyPrices><sellPrices>{asks}</sellPrices></pitch>"
    )


def envelope(*pitches):
    return "<envelope><message><market><pitches>" + "".join(pitches) + (
        "</pitches></market></message></envelope>"
    )


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """A regression in the transport setup must never contact a real account."""
    def blocked(*args, **kwargs):
        raise AssertionError("Real network access is forbidden in platinum tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def make_provider():
    clients = []

    def make(xml=None, handler=None, session_cookie=True, **config):
        calls = []
        text = envelope(pitch()) if xml is None else xml

        def dispatch(request):
            calls.append(request)
            if handler is not None:
                response = handler(request)
            else:
                assert request.method == "GET"
                assert request.url.path == "/view_market_xml.do"
                response = httpx.Response(200, text=text)
            if request.url.path == "/secure/login.do" and session_cookie:
                response.headers.setdefault(
                    "Set-Cookie", "JSESSIONID=synthetic-session; Path=/; Secure; HttpOnly"
                )
            return response

        # Deliberately unsafe client default: the adapter must override each request.
        client = httpx.Client(transport=httpx.MockTransport(dispatch), follow_redirects=True)
        clients.append(client)
        kwargs = {"security_id": "PTXLN", "access_mode": "public"}
        kwargs.update(config)
        return BullionVaultQuoteProvider(client=client, **kwargs), calls

    yield make
    for client in clients:
        client.close()


def test_platinum_public_quote_with_synthetic_fixture(make_provider):
    provider, calls = make_provider()
    before = datetime.now(timezone.utc)
    quote = provider.fetch_quote()
    after = datetime.now(timezone.utc)
    assert quote.security_id == "PTXLN"
    assert quote.currency == "USD"
    assert quote.source_provider == "BullionVault"
    assert quote.best_bid_usd_per_kg == 50000
    assert quote.best_ask_usd_per_kg == 50100
    assert quote.best_bid_quantity_kg == 0.01
    assert quote.best_ask_quantity_kg == 0.02
    assert quote.mid_usd_per_kg == 50050
    assert quote.spread_usd_per_kg == 100
    assert before <= quote.fetched_at_utc <= after
    assert quote.fetched_at_utc.utcoffset() == timedelta(0)
    assert quote.access_mode == "PUBLIC_CACHED_READ_ONLY"
    assert quote.freshness_status == "SERVER_CACHED_LESS_CURRENT"
    assert len(calls) == 1
    assert dict(calls[0].url.params) == {
        "securityId": "PTXLN", "considerationCurrency": "USD",
        "quantity": "0.001", "marketWidth": "5",
    }
    data = quote.as_dict()
    assert data["read_only"] is True
    assert data["execution_guaranteed"] is False
    assert data["bid_depth"][0] == {"price_usd_per_kg": 50000.0, "quantity_kg": 0.01}
    for name in ("place_order", "cancel_order", "buy", "sell", "withdraw"):
        assert not hasattr(provider, name)


@pytest.mark.parametrize("mode", ["public", "auto"])
def test_no_credentials_uses_public_market(make_provider, mode):
    provider, calls = make_provider(access_mode=mode)
    assert provider.resolved_access_mode == "public"
    assert provider.fetch_quote().access_mode == "PUBLIC_CACHED_READ_ONLY"
    assert [r.method for r in calls] == ["GET"]


def test_public_mode_never_sends_configured_credentials(make_provider):
    provider, calls = make_provider(username="synthetic-user", password="synthetic-password")
    provider.fetch_quote()
    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert "synthetic" not in str(calls[0].url)
    assert calls[0].content == b""


@pytest.mark.parametrize("security", ["PTXLN", "ptxln", " PTXLN "])
def test_platinum_symbol_normalization(make_provider, security):
    provider, _ = make_provider(security_id=security, currency=" usd ")
    assert provider.fetch_quote().security_id == "PTXLN"


@pytest.mark.parametrize("security", ["AGXLN", "AGXZU", "AGXTR", "AGXSG"])
def test_all_existing_silver_markets_still_work(make_provider, security):
    provider, calls = make_provider(envelope(pitch(security=security)), security_id=security)
    assert provider.fetch_quote().security_id == security
    assert calls[0].url.params["securityId"] == security


def test_default_market_remains_silver(make_provider):
    provider, _ = make_provider(envelope(pitch(security="AGXLN")), security_id="")
    assert provider.fetch_quote().security_id == "AGXLN"
    assert LiveSettings().bullionvault_security_id == "AGXLN"


def test_live_settings_does_not_enable_platinum_forecasting():
    # Important scope guard: the quote adapter is not the silver model pipeline.
    with pytest.raises(ValueError, match="Silver security"):
        LiveSettings(bullionvault_security_id="PTXLN")


@pytest.mark.parametrize("config, message", [
    ({"security_id": "UNKNOWN"}, "security"),
    ({"currency": "EUR"}, "requires USD"),
    ({"access_mode": "invalid"}, "access_mode"),
    ({"username": "synthetic-user"}, "configured together"),
    ({"password": "synthetic-password"}, "configured together"),
    ({"access_mode": "authenticated"}, "requires username and password"),
    *[({"minimum_quantity_kg": v}, "finite and positive")
      for v in (0, -1, float("nan"), float("inf"), -float("inf"))],
    *[({"market_width": v}, "between 1 and 20") for v in (0, -1, 21)],
])
def test_invalid_configuration_rejected_before_request(make_provider, config, message):
    with pytest.raises(ValueError, match=message):
        make_provider(**config)


@pytest.mark.parametrize("width", [1, 20])
def test_market_filter_boundaries_forwarded(make_provider, width):
    provider, calls = make_provider(market_width=width, minimum_quantity_kg=0.025)
    provider.fetch_quote()
    assert calls[0].url.params["marketWidth"] == str(width)
    assert float(calls[0].url.params["quantity"]) == 0.025


def test_selects_platinum_usd_from_multiple_markets(make_provider):
    xml = envelope(
        pitch("AGXLN"), pitch(currency="EUR"),
        pitch("PTXZU"), pitch(),
    )
    provider, _ = make_provider(xml)
    assert provider.fetch_quote().security_id == "PTXLN"


@pytest.mark.parametrize("xml", [
    envelope(pitch("AGXLN")), envelope(pitch(currency="EUR")),
    envelope(pitch("PTXZU")), "<envelope/>", "<html><body>login</body></html>",
])
def test_wrong_or_missing_market_fails_closed(make_provider, xml):
    provider, _ = make_provider(xml)
    with pytest.raises(RuntimeError, match="requested metal market"):
        provider.fetch_quote()


@pytest.mark.parametrize("xml, message", [
    ("", "invalid market XML"), ("<envelope>", "invalid market XML"),
    ('<!DOCTYPE envelope><envelope/>', "forbidden declarations"),
    ('<!ENTITY x "synthetic"><envelope/>', "forbidden declarations"),
    ('<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///not-a-real-secret">]><envelope/>',
     "forbidden declarations"),
])
def test_malformed_or_unsafe_xml_rejected(make_provider, xml, message):
    provider, _ = make_provider(xml)
    with pytest.raises(RuntimeError, match=message):
        provider.fetch_quote()


def test_depth_sorted_with_quantities_attached(make_provider):
    provider, _ = make_provider(envelope(pitch(
        bids='<price limit="49900" quantity="0.3"/><price limit="50000" quantity="0.1"/>',
        asks='<price limit="50200" quantity="0.4"/><price limit="50100" quantity="0.2"/>',
    )))
    quote = provider.fetch_quote()
    assert [(x.price_usd_per_kg, x.quantity_kg) for x in quote.bid_depth] == [
        (50000, 0.1), (49900, 0.3),
    ]
    assert [(x.price_usd_per_kg, x.quantity_kg) for x in quote.ask_depth] == [
        (50100, 0.2), (50200, 0.4),
    ]


@pytest.mark.parametrize("side", ["bids", "asks"])
@pytest.mark.parametrize("level", [
    '<price/>', '<price quantity="1"/>', '<price limit="50000"/>',
    *[f'<price limit="{v}" quantity="0.1"/>'
      for v in ("bad", "NaN", "inf", "-inf", "1e9999", "0", "-1")],
    *[f'<price limit="50050" quantity="{v}"/>'
      for v in ("bad", "NaN", "inf", "-inf", "1e9999", "0", "-1")],
])
def test_invalid_levels_skipped_without_corrupting_valid_book(make_provider, side, level):
    valid = '<price quantity="0.01" limit="50000"/>' if side == "bids" else (
        '<price quantity="0.02" limit="50100"/>'
    )
    provider, _ = make_provider(envelope(pitch(**{side: level + valid})))
    quote = provider.fetch_quote()
    assert len(quote.bid_depth) == len(quote.ask_depth) == 1
    assert quote.mid_usd_per_kg == 50050


@pytest.mark.parametrize("side", ["bids", "asks"])
@pytest.mark.parametrize("levels", ["", '<price quantity="0" limit="NaN"/>'])
def test_empty_or_all_invalid_side_rejected(make_provider, side, levels):
    provider, _ = make_provider(envelope(pitch(**{side: levels})))
    with pytest.raises(RuntimeError, match="no usable bid/ask depth"):
        provider.fetch_quote()


@pytest.mark.parametrize("ask", [50000, 49999])
def test_locked_or_crossed_book_rejected(make_provider, ask):
    provider, _ = make_provider(envelope(pitch(asks=f'<price limit="{ask}" quantity="1"/>')))
    with pytest.raises(RuntimeError, match="crossed or invalid"):
        provider.fetch_quote()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_public_http_errors_sanitized(make_provider, status):
    provider, _ = make_provider(handler=lambda r: httpx.Response(status, text="synthetic-secret"))
    with pytest.raises(RuntimeError, match=f"public market request failed with HTTP {status}") as error:
        provider.fetch_quote()
    assert "synthetic-secret" not in str(error.value)


def test_public_timeout_sanitized(make_provider):
    def handler(request):
        raise httpx.ReadTimeout("synthetic-secret", request=request)

    provider, _ = make_provider(handler=handler)
    with pytest.raises(RuntimeError, match="public market transport failed") as error:
        provider.fetch_quote()
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize("mode", ["auto", "authenticated"])
def test_authenticated_session_reused_and_only_login_posts(make_provider, mode):
    def handler(request):
        if request.url.path.endswith("view_market_xml.do"):
            return httpx.Response(200, text=envelope(pitch()))
        return httpx.Response(200, text="<html>synthetic login result</html>")

    provider, calls = make_provider(
        handler=handler, access_mode=mode,
        username="synthetic-user", password="synthetic-password",
    )
    for _ in range(2):
        quote = provider.fetch_quote()
        assert quote.access_mode == "AUTHENTICATED_READ_ONLY"
        assert quote.freshness_status == "CURRENT_GUI_SOURCE"
    assert [(r.method, r.url.path) for r in calls] == [
        ("GET", "/secure/login.do"), ("POST", "/secure/j_security_check"),
        ("GET", "/secure/api/v2/view_market_xml.do"),
        ("GET", "/secure/api/v2/view_market_xml.do"),
    ]
    assert parse_qs(calls[1].content.decode()) == {
        "j_username": ["synthetic-user"], "j_password": ["synthetic-password"],
    }
    for request in (calls[2], calls[3]):
        assert request.url.params["securityId"] == "PTXLN"
        assert "synthetic-password" not in str(request.url)


@pytest.mark.parametrize("failure_path", [
    "/secure/login.do", "/secure/j_security_check", "/secure/api/v2/view_market_xml.do",
])
@pytest.mark.parametrize("failure_type", ["http", "transport"])
@pytest.mark.parametrize("mode, fallback, allowed", [
    ("auto", True, True), ("auto", False, False), ("authenticated", True, False),
])
def test_auth_failures_obey_fallback_policy(
    make_provider, failure_path, failure_type, mode, fallback, allowed,
):
    def handler(request):
        if request.url.path == failure_path:
            if failure_type == "transport":
                raise httpx.ConnectError("synthetic-secret", request=request)
            return httpx.Response(401, text="synthetic-secret")
        return httpx.Response(200, text=envelope(pitch()))

    provider, calls = make_provider(
        handler=handler, access_mode=mode, allow_public_fallback=fallback,
        username="synthetic-user", password="synthetic-password",
    )
    if allowed:
        quote = provider.fetch_quote()
        assert quote.access_mode == "PUBLIC_CACHED_FALLBACK"
        assert quote.freshness_status == "SERVER_CACHED_LESS_CURRENT"
        assert calls[-1].url.path == "/view_market_xml.do"
    else:
        with pytest.raises(RuntimeError) as error:
            provider.fetch_quote()
        assert "synthetic-secret" not in str(error.value)
        assert all(r.url.path != "/view_market_xml.do" for r in calls)


def test_authenticated_market_failure_requires_new_login(make_provider):
    failed_once = False

    def handler(request):
        nonlocal failed_once
        if request.url.path == "/secure/api/v2/view_market_xml.do" and not failed_once:
            failed_once = True
            return httpx.Response(401)
        return httpx.Response(200, text=envelope(pitch()))

    provider, calls = make_provider(
        handler=handler, access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    with pytest.raises(RuntimeError, match="HTTP 401"):
        provider.fetch_quote()
    assert provider.fetch_quote().access_mode == "AUTHENTICATED_READ_ONLY"
    assert sum(r.url.path == "/secure/j_security_check" for r in calls) == 2


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("location", [
    "https://redirect-sink.invalid/login",
    "http://www.bullionvault.com/secure/home.do",
    "/secure/home.do",
])
def test_login_never_replays_credential_post(make_provider, status, location):
    def handler(request):
        if request.url.path == "/secure/j_security_check":
            return httpx.Response(status, headers={"Location": location})
        return httpx.Response(200, text=envelope(pitch()))

    provider, calls = make_provider(
        handler=handler, access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    with pytest.raises(RuntimeError):
        provider.fetch_quote()
    assert len(calls) == 2
    assert all(r.url.host == "www.bullionvault.com" for r in calls)
    assert sum(r.method == "POST" for r in calls) == 1


@pytest.mark.parametrize("status", [302, 303])
@pytest.mark.parametrize("location", [
    "/secure/home.do", "https://www.bullionvault.com/secure/home.do",
])
def test_safe_login_redirect_not_followed_before_secure_read(make_provider, status, location):
    def handler(request):
        if request.url.path == "/secure/j_security_check":
            return httpx.Response(status, headers={"Location": location})
        return httpx.Response(200, text=envelope(pitch()))

    provider, calls = make_provider(
        handler=handler, access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    assert provider.fetch_quote().security_id == "PTXLN"
    assert len(calls) == 3
    assert all(r.url.path != "/secure/home.do" for r in calls)


@pytest.mark.parametrize("status", [302, 303])
@pytest.mark.parametrize("location", [
    "", "https://redirect-sink.invalid/secure/home.do",
    "//redirect-sink.invalid/secure/home.do",
    "http://www.bullionvault.com/secure/home.do",
    "https://www.bullionvault.com:444/secure/home.do",
    "https://synthetic:password@www.bullionvault.com/secure/home.do",
    "/view_market_xml.do", "/secure/login.do", "/secure/j_security_check",
    "https://www.bullionvault.com.redirect-sink.invalid/secure/home.do",
])
def test_unsafe_login_acknowledgement_rejected(make_provider, status, location):
    def handler(request):
        if request.url.path == "/secure/j_security_check":
            return httpx.Response(status, headers={"Location": location})
        return httpx.Response(200, text=envelope(pitch()))

    provider, calls = make_provider(
        handler=handler, access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    with pytest.raises(RuntimeError, match="login redirect"):
        provider.fetch_quote()
    assert len(calls) == 2


@pytest.mark.parametrize("mode, allowed", [
    ("authenticated", False), ("auto", True),
])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("location", [
    "/view_market_xml.do", "/secure/login.do",
    "https://redirect-sink.invalid/view_market_xml.do",
])
def test_secure_market_redirect_never_labelled_authenticated(
    make_provider, mode, allowed, status, location,
):
    def handler(request):
        if request.url.path == "/secure/api/v2/view_market_xml.do":
            return httpx.Response(status, headers={"Location": location})
        return httpx.Response(200, text=envelope(pitch()))

    provider, calls = make_provider(
        handler=handler, access_mode=mode,
        username="synthetic-user", password="synthetic-password",
    )
    if allowed:
        quote = provider.fetch_quote()
        assert quote.access_mode == "PUBLIC_CACHED_FALLBACK"
        assert quote.freshness_status == "SERVER_CACHED_LESS_CURRENT"
        assert calls[-1].url.path == "/view_market_xml.do"
    else:
        with pytest.raises(RuntimeError):
            provider.fetch_quote()
        assert calls[-1].url.path == "/secure/api/v2/view_market_xml.do"
    assert all(r.url.host == "www.bullionvault.com" for r in calls)


@pytest.mark.parametrize("session_cookie", [True, False])
@pytest.mark.parametrize("secure_body", [
    "<html><body>Login failed</body></html>", "<envelope/>", "<broken>",
])
def test_login_http_200_not_sufficient_and_next_attempt_logs_in_again(
    make_provider, session_cookie, secure_body,
):
    def handler(request):
        if request.url.path == "/secure/api/v2/view_market_xml.do":
            return httpx.Response(200, text=secure_body)
        return httpx.Response(200, text="<html>Login form</html>")

    provider, calls = make_provider(
        handler=handler, session_cookie=session_cookie, access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            provider.fetch_quote()
    assert sum(r.url.path == "/secure/j_security_check" for r in calls) == 2


def test_secure_xml_without_session_cookie_is_rejected(make_provider):
    provider, _ = make_provider(
        handler=lambda r: httpx.Response(200, text=envelope(pitch())),
        session_cookie=False, access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    with pytest.raises(RuntimeError, match="session cookie"):
        provider.fetch_quote()


@pytest.mark.parametrize("bad_url", [
    "https://www.bullionvault.com/view_market_xml.do",
    "https://redirect-sink.invalid/secure/api/v2/view_market_xml.do",
    "http://www.bullionvault.com/secure/api/v2/view_market_xml.do",
])
def test_injected_client_cannot_forge_secure_response_location(make_provider, bad_url):
    base_provider, _ = make_provider(
        handler=lambda r: httpx.Response(200, text=envelope(pitch())),
    )

    class ResponseSwappingClient:
        def get(self, url, **kwargs):
            response = base_provider._client.get(url, **kwargs)
            if url.endswith("/secure/api/v2/view_market_xml.do"):
                # Change it after httpx binds its request to the transport response.
                response.request = httpx.Request("GET", bad_url)
            return response

        def post(self, url, **kwargs):
            return base_provider._client.post(url, **kwargs)

    provider = BullionVaultQuoteProvider(
        client=ResponseSwappingClient(), security_id="PTXLN", access_mode="authenticated",
        username="synthetic-user", password="synthetic-password",
    )
    with pytest.raises(RuntimeError, match="response target"):
        provider.fetch_quote()


@pytest.mark.parametrize("age_seconds, error", [
    (0, None), (180, None), (181, "stale"), (-6, "unexpectedly in the future"),
])
def test_platinum_snapshot_roundtrip_preserves_identity_and_freshness(
    make_provider, tmp_path, age_seconds, error,
):
    quote = make_provider()[0].fetch_quote()
    repository = SQLiteMicrostructureRepository(tmp_path / "platinum-only.sqlite3")
    snapshot = MicrostructureSnapshot.from_quote(quote)
    repository.append(snapshot, MicrostructureFeatureBuilder().build(snapshot, None))
    replay = PersistedMicrostructureQuoteProvider(
        repository, max_age_seconds=180,
        clock=lambda: quote.fetched_at_utc + timedelta(seconds=age_seconds),
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            replay.fetch_quote()
        assert replay.status()["stale"] is True
    else:
        result = replay.fetch_quote()
        assert result.as_dict() == quote.as_dict()
        assert result.security_id == "PTXLN"
        assert result.freshness_status == "SERVER_CACHED_LESS_CURRENT"
    assert replay.status()["direct_network_request"] is False
