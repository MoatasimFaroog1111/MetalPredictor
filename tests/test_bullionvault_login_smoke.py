"""Synthetic-only safety tests. Never use environment or real account credentials."""
import json
import logging
import socket
import warnings
from urllib.parse import urlencode

import httpx2 as httpx
import pytest

from metal_predictor.live import bullionvault_login_smoke as smoke

XML = """<envelope><pitch securityId="PTXLN" considerationCurrency="USD">
<buyPrices><price limit="50000" quantity="1"/></buyPrices>
<sellPrices><price limit="50100" quantity="1"/></sellPrices></pitch></envelope>"""
LIVE = ["--live", "--confirm-read-only-login", "--credentials-from-env"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No real sockets allowed")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    # Deterministic fixture credentials only; no existing secrets are inspected.
    monkeypatch.setenv("BULLIONVAULT_USERNAME", "synthetic-user")
    monkeypatch.setenv("BULLIONVAULT_PASSWORD", "synthetic-password")


def response(request):
    if request.url.path == "/secure/login.do":
        return httpx.Response(
            200, text="<html>login</html>",
            headers={"Set-Cookie": "JSESSIONID=synthetic-session; Secure; Path=/; HttpOnly"},
        )
    return httpx.Response(200, text=XML)


def test_default_preflight_never_reads_credentials_or_creates_transport(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not load secrets or initialize network")
    monkeypatch.setattr(smoke, "read_credentials", forbidden)
    monkeypatch.setattr(smoke.httpx, "HTTPTransport", forbidden)
    assert smoke.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "READY_NOT_RUN"
    assert result["live_login_verified"] is False
    assert result["network_requests"] == 0


@pytest.mark.parametrize("args", [
    ["--live"], ["--credentials-from-env"], ["--confirm-read-only-login"],
    ["--password", "synthetic-password"], ["--security-id", "synthetic-user"],
])
def test_invalid_or_partial_opt_in_never_loads_secrets(args, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "read_credentials", lambda *a: pytest.fail("No credential access"))
    assert smoke.main(args) == 2
    output = capsys.readouterr()
    assert "synthetic" not in output.out + output.err


def test_one_shot_success_has_three_requests_and_no_sensitive_output(capsys):
    calls = []
    def handler(request):
        calls.append(request)
        return response(request)
    result = smoke.run_live_smoke(
        "synthetic-user", "synthetic-password", transport=httpx.MockTransport(handler)
    )
    assert result["status"] == "PASS"
    assert result["network_requests"] == 3
    assert [(r.method, r.url.path) for r in calls] == list(smoke.SEQUENCE)
    assert "JSESSIONID=synthetic-session" in calls[-1].headers["cookie"]
    assert calls[-1].headers["accept-encoding"] == "identity"
    assert dict(calls[-1].url.params)["securityId"] == "PTXLN"
    assert not any("synthetic" in str(value) for value in result.values())
    assert capsys.readouterr().out == ""


def test_cli_success_outputs_metadata_only(monkeypatch, capsys):
    original = smoke.run_live_smoke
    monkeypatch.setattr(
        smoke, "run_live_smoke",
        lambda u, p, s: original(u, p, s, transport=httpx.MockTransport(response)),
    )
    assert smoke.main(LIVE) == 0
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert data["live_login_verified"] is True
    for private in ("synthetic", "50000", "50100", "JSESSIONID", "password"):
        assert private not in output.out + output.err


@pytest.mark.parametrize("url,method", [
    ("https://www.bullionvault.com/secure/api/v2/place_order_xml.do", "POST"),
    ("https://www.bullionvault.com/secure/api/v2/cancel_order_xml.do", "POST"),
    ("https://www.bullionvault.com/secure/api/v2/view_balance_xml.do", "GET"),
    ("https://www.bullionvault.com/view_market_xml.do", "GET"),
    ("https://redirect-sink.invalid/secure/login.do", "GET"),
    ("http://www.bullionvault.com/secure/login.do", "GET"),
    ("https://www.bullionvault.com:444/secure/login.do", "GET"),
    ("https://synthetic:password@www.bullionvault.com/secure/login.do", "GET"),
    ("https://www.bullionvault.com/secure/%6cogin.do", "GET"),
    ("https://www.bullionvault.com/secure/login.do?password=synthetic", "GET"),
    ("https://www.bullionvault.com/secure/login.do#fragment", "GET"),
    ("https://www.bullionvault.com/secure/login.do", "DELETE"),
])
def test_policy_blocks_disallowed_requests_before_transport(url, method):
    guard = smoke.LoginOnlyTransport(
        httpx.MockTransport(lambda r: pytest.fail("Request must not be sent")), "PTXLN"
    )
    with pytest.raises(smoke.SmokeSafetyError):
        guard.handle_request(httpx.Request(method, url))
    assert guard.requests_sent == 0


@pytest.mark.parametrize("fields", [
    {"j_username": "synthetic"}, {"j_password": "synthetic"},
    {"j_username": "synthetic", "j_password": "synthetic", "order": "buy"},
    [("j_username", "synthetic"), ("j_username", "duplicate"), ("j_password", "synthetic")],
    [("j_username", ""), ("j_username", "synthetic"), ("j_password", "synthetic")],
])
def test_policy_rejects_non_login_form_or_duplicate_fields(fields):
    guard = smoke.LoginOnlyTransport(
        httpx.MockTransport(lambda r: pytest.fail("No send")), "PTXLN"
    )
    guard.requests_sent = 1
    with pytest.raises(smoke.SmokeSafetyError):
        guard.handle_request(httpx.Request(
            "POST", f"https://{smoke.HOST}/secure/j_security_check",
            content=urlencode(fields), headers={"Content-Type": "application/x-www-form-urlencoded"},
        ))


@pytest.mark.parametrize("query", [
    "securityId=AGXLN&considerationCurrency=USD&quantity=0.001&marketWidth=1",
    "securityId=PTXLN&considerationCurrency=USD&quantity=0.001&marketWidth=1&order=buy",
    "securityId=PTXLN&securityId=PTXLN&considerationCurrency=USD&quantity=0.001&marketWidth=1",
])
def test_policy_rejects_wrong_market_filters(query):
    guard = smoke.LoginOnlyTransport(
        httpx.MockTransport(lambda r: pytest.fail("No send")), "PTXLN"
    )
    guard.requests_sent = 2
    with pytest.raises(smoke.SmokeSafetyError):
        guard.handle_request(httpx.Request(
            "GET", f"https://{smoke.HOST}/secure/api/v2/view_market_xml.do?{query}"
        ))


def test_request_budget_blocks_a_fourth_request():
    guard = smoke.LoginOnlyTransport(
        httpx.MockTransport(lambda r: pytest.fail("No send")), "PTXLN"
    )
    guard.requests_sent = 3
    with pytest.raises(smoke.SmokeSafetyError, match="BUDGET"):
        guard.handle_request(httpx.Request("GET", f"https://{smoke.HOST}/secure/login.do"))


@pytest.mark.parametrize("failure", ["http", "timeout", "redirect", "html", "cookie"])
def test_failed_login_never_retries_falls_back_or_leaks(failure, monkeypatch, capsys):
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path == "/secure/j_security_check":
            if failure == "http":
                return httpx.Response(401, text="synthetic-password")
            if failure == "timeout":
                raise httpx.ReadTimeout("synthetic-password", request=request)
            if failure == "redirect":
                return httpx.Response(
                    307, headers={"Location": "https://redirect-sink.invalid/secure/login.do"}
                )
        if failure == "cookie":
            return httpx.Response(200, text=XML)
        if failure == "html" and request.url.path.endswith("view_market_xml.do"):
            return httpx.Response(200, text="<html>synthetic-password</html>")
        return response(request)
    original = smoke.run_live_smoke
    monkeypatch.setattr(
        smoke, "run_live_smoke",
        lambda u, p, s: original(u, p, s, transport=httpx.MockTransport(handler)),
    )
    assert smoke.main(LIVE) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "FAIL"
    assert len(calls) <= 3
    assert sum(r.method == "POST" for r in calls) == 1
    assert all(r.url.host == smoke.HOST for r in calls)
    assert all(r.url.path != "/view_market_xml.do" for r in calls)
    assert "synthetic" not in output.out + output.err
    assert "Traceback" not in output.out + output.err


@pytest.mark.parametrize("encoding", ["gzip", "br"])
def test_compressed_responses_fail_closed(encoding):
    with pytest.raises(smoke.SmokeSafetyError, match="COMPRESSED"):
        smoke.run_live_smoke(
            "synthetic", "synthetic",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, headers={"Content-Encoding": encoding}, content=b"")
            ),
        )


def test_oversized_response_aborts_before_credentials_post():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=b"x" * (smoke.MAX_RESPONSE_BYTES + 1))
    with pytest.raises(smoke.SmokeSafetyError, match="TOO_LARGE"):
        smoke.run_live_smoke("synthetic", "synthetic", transport=httpx.MockTransport(handler))
    assert len(calls) == 1


def test_streamed_response_has_same_byte_limit_and_is_closed():
    class OversizedStream(httpx.SyncByteStream):
        closed = False
        def __iter__(self):
            yield b"x" * smoke.MAX_RESPONSE_BYTES
            yield b"x"
        def close(self):
            self.closed = True
    stream = OversizedStream()
    with pytest.raises(smoke.SmokeSafetyError, match="TOO_LARGE"):
        smoke.run_live_smoke(
            "synthetic", "synthetic",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)),
        )
    assert stream.closed is True


def test_nonterminal_input_never_falls_back_to_echo(monkeypatch):
    monkeypatch.setattr(smoke.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(smoke.getpass, "getpass", lambda *a: pytest.fail("No unsafe prompt"))
    with pytest.raises(smoke.SmokeInputError, match="TERMINAL"):
        smoke.read_credentials(False)


def test_missing_environment_values_are_not_reported(monkeypatch, capsys):
    monkeypatch.delenv("BULLIONVAULT_PASSWORD")
    assert smoke.main(LIVE) == 2
    assert "synthetic-user" not in capsys.readouterr().out


def test_http_logs_are_suppressed_and_global_state_restored():
    previous = logging.root.manager.disable
    def handler(request):
        assert logging.root.manager.disable == logging.CRITICAL
        return response(request)
    smoke.run_live_smoke("synthetic", "synthetic", transport=httpx.MockTransport(handler))
    assert logging.root.manager.disable == previous


def test_real_transport_defaults_enable_tls_and_disable_proxy_env_and_retries(monkeypatch):
    real_client = httpx.Client
    options = {}
    def transport_factory(**kwargs):
        options["transport"] = kwargs
        return httpx.MockTransport(response)
    def client_factory(**kwargs):
        options["client"] = kwargs
        return real_client(**kwargs)
    monkeypatch.setattr(smoke.httpx, "HTTPTransport", transport_factory)
    monkeypatch.setattr(smoke.httpx, "Client", client_factory)
    assert smoke.run_live_smoke("synthetic", "synthetic")["status"] == "PASS"
    assert options["transport"]["verify"] is True
    assert options["transport"]["trust_env"] is False
    assert options["transport"]["retries"] == 0
    assert options["client"]["trust_env"] is False
    assert options["client"]["follow_redirects"] is False
    assert options["client"]["timeout"].read == 10


def test_unsafe_getpass_fallback_fails_before_password_is_echoed(monkeypatch, capsys):
    monkeypatch.setattr(smoke.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(smoke.sys.stderr, "isatty", lambda: True)
    def unsafe_prompt(*args):
        warnings.warn("synthetic-password", smoke.getpass.GetPassWarning)
        pytest.fail("Must stop before falling back to echo")
    monkeypatch.setattr(smoke.getpass, "getpass", unsafe_prompt)
    assert smoke.main(["--live", "--confirm-read-only-login"]) == 2
    output = capsys.readouterr()
    assert "synthetic" not in output.out + output.err


def test_logs_restored_after_failure():
    previous = logging.root.manager.disable
    with pytest.raises(RuntimeError):
        smoke.run_live_smoke(
            "synthetic", "synthetic",
            transport=httpx.MockTransport(lambda r: httpx.Response(401)),
        )
    assert logging.root.manager.disable == previous


def test_user_cancellation_is_sanitized(monkeypatch, capsys):
    def cancelled(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(smoke, "read_credentials", cancelled)
    assert smoke.main(LIVE) == 130
    assert json.loads(capsys.readouterr().out)["reason"] == "USER_CANCELLED"
