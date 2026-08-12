from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any
import xml.etree.ElementTree as ET

import httpx2 as httpx

from metal_predictor.live.quote_contracts import (
    BidAskMarketQuote,
    MarketDepthLevel,
)


class BullionVaultQuoteProvider:
    """Read-only BullionVault Silver market-depth adapter.

    This adapter deliberately implements only BullionVault's view-market API. It has no
    place-order, cancel-order, balance, or account mutation methods. Authenticated access
    is preferred because BullionVault documents the public view-market response as
    server-cached and less current than the logged-in market view.
    """

    _BASE_URL = "https://www.bullionvault.com"
    _PUBLIC_MARKET_PATH = "/view_market_xml.do"
    _SECURE_MARKET_PATH = "/secure/api/v2/view_market_xml.do"
    _LOGIN_PATH = "/secure/login.do"
    _SECURITY_CHECK_PATH = "/secure/j_security_check"
    _SILVER_SECURITIES = {"AGXLN", "AGXZU", "AGXTR", "AGXSG"}

    def __init__(
        self,
        *,
        username: str = "",
        password: str = "",
        security_id: str = "AGXLN",
        currency: str = "USD",
        minimum_quantity_kg: float = 0.001,
        market_width: int = 5,
        access_mode: str = "auto",
        allow_public_fallback: bool = True,
        client: Any | None = None,
    ) -> None:
        self._username = username.strip()
        self._password = password
        self._security_id = security_id.strip().upper() or "AGXLN"
        self._currency = currency.strip().upper() or "USD"
        self._minimum_quantity_kg = float(minimum_quantity_kg)
        self._market_width = int(market_width)
        self._access_mode = access_mode.strip().lower() or "auto"
        self._allow_public_fallback = bool(allow_public_fallback)
        self._client = client or httpx.Client(timeout=20.0, follow_redirects=True)
        self._authenticated = False
        self._validate_configuration()

    @property
    def resolved_access_mode(self) -> str:
        if self._access_mode == "public":
            return "public"
        if self._access_mode == "authenticated":
            return "authenticated"
        return "authenticated" if self._has_credentials else "public"

    @property
    def security_id(self) -> str:
        return self._security_id

    def fetch_quote(self) -> BidAskMarketQuote:
        mode = self.resolved_access_mode
        if mode == "authenticated":
            try:
                return self._fetch_authenticated()
            except RuntimeError:
                if self._access_mode == "authenticated" or not self._allow_public_fallback:
                    raise
                return self._fetch_public(fallback=True)
        return self._fetch_public(fallback=False)

    @property
    def _has_credentials(self) -> bool:
        return bool(self._username and self._password)

    def _validate_configuration(self) -> None:
        if self._access_mode not in {"auto", "authenticated", "public"}:
            raise ValueError("BullionVault access_mode must be auto, authenticated, or public.")
        if bool(self._username) != bool(self._password):
            raise ValueError("BullionVault username and password must be configured together.")
        if self._access_mode == "authenticated" and not self._has_credentials:
            raise ValueError("Authenticated BullionVault mode requires username and password.")
        if self._security_id not in self._SILVER_SECURITIES:
            raise ValueError("BullionVault Silver security must be AGXLN, AGXZU, AGXTR, or AGXSG.")
        if self._currency != "USD":
            raise ValueError("MetalPredictor BullionVault quote integration currently requires USD.")
        if not math.isfinite(self._minimum_quantity_kg) or self._minimum_quantity_kg <= 0:
            raise ValueError("BullionVault minimum quantity must be finite and positive.")
        if not 1 <= self._market_width <= 20:
            raise ValueError("BullionVault market width must be between 1 and 20.")

    def _fetch_authenticated(self) -> BidAskMarketQuote:
        self._ensure_authenticated()
        try:
            response = self._client.get(
                f"{self._BASE_URL}{self._SECURE_MARKET_PATH}",
                params=self._market_params(),
            )
            response.raise_for_status()
            return self._parse_market_response(
                response.text,
                access_mode="AUTHENTICATED_READ_ONLY",
                freshness_status="CURRENT_GUI_SOURCE",
            )
        except httpx.HTTPStatusError as exc:
            self._authenticated = False
            raise RuntimeError(
                f"BullionVault authenticated market request failed with HTTP {exc.response.status_code}."
            ) from None
        except httpx.HTTPError:
            self._authenticated = False
            raise RuntimeError("BullionVault authenticated market transport failed.") from None

    def _fetch_public(self, *, fallback: bool) -> BidAskMarketQuote:
        try:
            response = self._client.get(
                f"{self._BASE_URL}{self._PUBLIC_MARKET_PATH}",
                params=self._market_params(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"BullionVault public market request failed with HTTP {exc.response.status_code}."
            ) from None
        except httpx.HTTPError:
            raise RuntimeError("BullionVault public market transport failed.") from None

        return self._parse_market_response(
            response.text,
            access_mode=("PUBLIC_CACHED_FALLBACK" if fallback else "PUBLIC_CACHED_READ_ONLY"),
            freshness_status="SERVER_CACHED_LESS_CURRENT",
        )

    def _ensure_authenticated(self) -> None:
        if self._authenticated:
            return
        if not self._has_credentials:
            raise RuntimeError("BullionVault authenticated quote access is not configured.")
        try:
            login_page = self._client.get(f"{self._BASE_URL}{self._LOGIN_PATH}")
            login_page.raise_for_status()
            response = self._client.post(
                f"{self._BASE_URL}{self._SECURITY_CHECK_PATH}",
                data={"j_username": self._username, "j_password": self._password},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"BullionVault login failed with HTTP {exc.response.status_code}."
            ) from None
        except httpx.HTTPError:
            raise RuntimeError("BullionVault login transport failed.") from None
        self._authenticated = True

    def _market_params(self) -> dict[str, object]:
        return {
            "considerationCurrency": self._currency,
            "securityId": self._security_id,
            "quantity": self._minimum_quantity_kg,
            "marketWidth": self._market_width,
        }

    def _parse_market_response(
        self,
        xml_text: str,
        *,
        access_mode: str,
        freshness_status: str,
    ) -> BidAskMarketQuote:
        text = str(xml_text)
        upper = text.upper()
        if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
            raise RuntimeError("BullionVault XML response contains forbidden declarations.")
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            raise RuntimeError("BullionVault returned invalid market XML.") from None

        pitch = None
        for candidate in root.findall(".//pitch"):
            if (
                candidate.attrib.get("securityId", "").upper() == self._security_id
                and candidate.attrib.get("considerationCurrency", "").upper() == self._currency
            ):
                pitch = candidate
                break
        if pitch is None:
            raise RuntimeError("BullionVault response did not contain the requested Silver market.")

        bids = self._parse_levels(pitch.findall("./buyPrices/price"), descending=True)
        asks = self._parse_levels(pitch.findall("./sellPrices/price"), descending=False)
        if not bids or not asks:
            raise RuntimeError("BullionVault market has no usable bid/ask depth for the requested filter.")
        best_bid = bids[0]
        best_ask = asks[0]
        if best_ask.price_usd_per_kg <= best_bid.price_usd_per_kg:
            raise RuntimeError("BullionVault returned a crossed or invalid top-of-book quote.")

        return BidAskMarketQuote(
            source_provider="BullionVault",
            security_id=self._security_id,
            currency=self._currency,
            best_bid_usd_per_kg=best_bid.price_usd_per_kg,
            best_ask_usd_per_kg=best_ask.price_usd_per_kg,
            best_bid_quantity_kg=best_bid.quantity_kg,
            best_ask_quantity_kg=best_ask.quantity_kg,
            bid_depth=tuple(bids),
            ask_depth=tuple(asks),
            fetched_at_utc=datetime.now(timezone.utc),
            access_mode=access_mode,
            freshness_status=freshness_status,
        )

    @staticmethod
    def _parse_levels(nodes: list[ET.Element], *, descending: bool) -> list[MarketDepthLevel]:
        levels: list[MarketDepthLevel] = []
        for node in nodes:
            try:
                price = float(node.attrib["limit"])
                quantity = float(node.attrib["quantity"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity <= 0:
                continue
            levels.append(MarketDepthLevel(price_usd_per_kg=price, quantity_kg=quantity))
        levels.sort(key=lambda level: level.price_usd_per_kg, reverse=descending)
        return levels
