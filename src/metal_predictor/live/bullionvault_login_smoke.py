"""Opt-in, bounded, read-only login smoke test; never run by CI against real accounts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import json
import logging
import os
import sys
import warnings
from urllib.parse import parse_qs

import httpx2 as httpx

from metal_predictor.live.bullionvault_quote import BullionVaultQuoteProvider

HOST = "www.bullionvault.com"
MARKETS = ("PTXLN", "AGXLN", "AGXZU", "AGXTR", "AGXSG")
MAX_RESPONSE_BYTES = 1024 * 1024
SEQUENCE = (
    ("GET", "/secure/login.do"),
    ("POST", "/secure/j_security_check"),
    ("GET", "/secure/api/v2/view_market_xml.do"),
)


class SmokeSafetyError(RuntimeError):
    """Fixed-message policy errors: never include requests, responses or secrets."""


class SmokeInputError(ValueError):
    pass


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes invalid arguments, possibly including a secret
        # pasted into an unsupported --password option. Never echo them.
        raise SmokeInputError("INVALID_ARGUMENTS")


class BoundedStream(httpx.SyncByteStream):
    def __init__(self, inner):
        self.inner = inner

    def __iter__(self):
        size = 0
        for chunk in self.inner:
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise SmokeSafetyError("RESPONSE_TOO_LARGE")
            yield chunk

    def close(self):
        self.inner.close()


class LoginOnlyTransport(httpx.BaseTransport):
    """Allow exactly one login attempt plus one secure quote, with no redirects.

    The guard is below the provider and HTTP client. Even a provider regression
    cannot call an order/balance endpoint or exceed the three-request sequence.
    """

    def __init__(self, inner, security_id):
        self.inner = inner
        self.security_id = security_id
        self.requests_sent = 0

    def handle_request(self, request):
        if self.requests_sent >= len(SEQUENCE):
            raise SmokeSafetyError("REQUEST_BUDGET_EXCEEDED")
        method, path = SEQUENCE[self.requests_sent]
        url = request.url
        if (
            request.method != method
            or url.scheme != "https"
            or url.host != HOST
            or url.port not in {None, 443}
            or url.userinfo
            or url.fragment
            or url.path != path
            or url.raw_path.split(b"?", 1)[0] != path.encode("ascii")
        ):
            raise SmokeSafetyError("REQUEST_NOT_ALLOWED")
        if method == "POST":
            if url.query:
                raise SmokeSafetyError("LOGIN_QUERY_NOT_ALLOWED")
            try:
                fields = parse_qs(
                    request.content.decode("utf-8"), strict_parsing=True, keep_blank_values=True
                )
            except (UnicodeError, ValueError):
                raise SmokeSafetyError("LOGIN_FORM_NOT_ALLOWED") from None
            if (
                request.headers.get("content-type", "").split(";")[0]
                != "application/x-www-form-urlencoded"
                or set(fields) != {"j_username", "j_password"}
                or any(len(values) != 1 or not values[0] for values in fields.values())
            ):
                raise SmokeSafetyError("LOGIN_FORM_NOT_ALLOWED")
        elif path == SEQUENCE[2][1]:
            expected = {
                "securityId": self.security_id, "considerationCurrency": "USD",
                "quantity": "0.001", "marketWidth": "1",
            }
            if sorted(request.url.params.multi_items()) != sorted(expected.items()):
                raise SmokeSafetyError("MARKET_FILTER_NOT_ALLOWED")
            if request.content:
                raise SmokeSafetyError("GET_BODY_NOT_ALLOWED")
        elif url.query or request.content:
            raise SmokeSafetyError("LOGIN_PAGE_QUERY_OR_BODY_NOT_ALLOWED")
        self.requests_sent += 1
        response = self.inner.handle_request(request)
        # Identity encoding plus a bounded raw stream also avoids compressed
        # payload expansion before XML parsing.
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            response.close()
            raise SmokeSafetyError("COMPRESSED_RESPONSE_NOT_ALLOWED")
        # Mock/custom transports can supply already-buffered content. Do not
        # let that bypass the same limit enforced while streaming real HTTP.
        if response.is_stream_consumed:
            try:
                size = len(response.content)
            except httpx.ResponseNotRead:
                response.close()
                raise SmokeSafetyError("UNEXPECTED_RESPONSE_STATE") from None
            if size > MAX_RESPONSE_BYTES:
                response.close()
                raise SmokeSafetyError("RESPONSE_TOO_LARGE")
        response.stream = BoundedStream(response.stream)
        return response

    def close(self):
        self.inner.close()


@contextmanager
def quiet_logs():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def run_live_smoke(username, password, security_id="PTXLN", *, transport=None):
    """One shot, no retries/fallback/persistence; transport injection is tests only."""
    if security_id not in MARKETS or not username.strip() or not password:
        raise SmokeInputError("INVALID_INPUT")
    with quiet_logs():
        inner = transport if transport is not None else httpx.HTTPTransport(
            verify=True, trust_env=False, retries=0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )
        guard = LoginOnlyTransport(inner, security_id)
        with httpx.Client(
            transport=guard, trust_env=False, verify=True, follow_redirects=False,
            timeout=httpx.Timeout(10.0),
            headers={"Accept-Encoding": "identity"},
        ) as client:
            try:
                provider = BullionVaultQuoteProvider(
                    username=username, password=password, security_id=security_id,
                    currency="USD", minimum_quantity_kg=0.001, market_width=1,
                    access_mode="authenticated", allow_public_fallback=False, client=client,
                )
                quote = provider.fetch_quote()
                if (
                    guard.requests_sent != 3
                    or quote.security_id != security_id
                    or quote.access_mode != "AUTHENTICATED_READ_ONLY"
                    or quote.freshness_status != "CURRENT_GUI_SOURCE"
                ):
                    raise SmokeSafetyError("AUTHENTICATION_NOT_VERIFIED")
                return {
                    "status": "PASS",
                    "live_login_verified": True,
                    "security_id": security_id,
                    "access_mode": quote.access_mode,
                    "network_requests": guard.requests_sent,
                    "order_endpoints_allowed": False,
                    "public_fallback_allowed": False,
                    "market_generation_time_verified": False,
                    "secrets_or_prices_in_output": False,
                }
            finally:
                client.cookies.clear()


def read_credentials(from_env):
    if from_env:
        username = os.environ.get("BULLIONVAULT_USERNAME", "")
        password = os.environ.get("BULLIONVAULT_PASSWORD", "")
    else:
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            raise SmokeInputError("SECURE_TERMINAL_REQUIRED")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            username = getpass.getpass("BullionVault username (hidden): ")
            password = getpass.getpass("BullionVault password (hidden): ")
    if not username.strip() or not password:
        raise SmokeInputError("CREDENTIALS_REQUIRED")
    return username, password


def main(argv=None):
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly allow one real login attempt")
    parser.add_argument("--confirm-read-only-login", action="store_true")
    parser.add_argument("--credentials-from-env", action="store_true",
                        help="Opt in to existing BULLIONVAULT_USERNAME/PASSWORD environment values")
    parser.add_argument("--security-id", choices=MARKETS, default="PTXLN")
    try:
        args = parser.parse_args(argv)
        if not args.live:
            if args.confirm_read_only_login or args.credentials_from_env:
                raise SmokeInputError("LIVE_FLAG_REQUIRED")
            result = {
                "status": "READY_NOT_RUN", "live_login_verified": False,
                "credentials_loaded": False, "network_requests": 0,
                "security_id": args.security_id, "order_endpoints_allowed": False,
            }
        else:
            if not args.confirm_read_only_login:
                raise SmokeInputError("EXPLICIT_CONFIRMATION_REQUIRED")
            username, password = read_credentials(args.credentials_from_env)
            result = run_live_smoke(username, password, args.security_id)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (SmokeInputError, getpass.GetPassWarning, EOFError):
        print(json.dumps({"status": "NOT_RUN", "reason": "INVALID_OR_MISSING_SAFE_INPUT"}))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "STOPPED", "reason": "USER_CANCELLED"}))
        return 130
    except Exception:
        # Deliberately do not serialize str(exc), repr(exc), tracebacks, headers,
        # cookies, request bodies, usernames, URLs or prices from a failed login.
        print(json.dumps({"status": "FAIL", "reason": "LIVE_CHECK_FAILED_NO_RETRY"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
