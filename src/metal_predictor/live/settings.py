from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path


def _environment_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


@dataclass(frozen=True)
class LiveSettings:
    repository_root: Path = Path(".")
    database_path: Path = Path("runtime/live_predictions.sqlite3")
    admin_token: str = ""
    market_provider: str = "auto"
    gold_api_key: str = ""
    gold_api_symbol: str = "XAG"
    twelvedata_api_key: str = ""
    twelvedata_symbol: str = "XAG/USD"
    auto_collect: bool = False
    collection_delay_minutes: int = 5
    bullionvault_username: str = ""
    bullionvault_password: str = ""
    bullionvault_access_mode: str = "auto"
    bullionvault_security_id: str = "AGXLN"
    bullionvault_currency: str = "USD"
    bullionvault_market_width: int = 5
    bullionvault_minimum_quantity_kg: float = 0.001
    bullionvault_public_fallback: bool = True
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = ""
    telegram_allowed_chat_ids: tuple[str, ...] = ()
    public_base_url: str = ""

    def __post_init__(self) -> None:
        provider = self.market_provider.strip().lower()
        if provider not in {"auto", "goldapi", "twelvedata"}:
            raise ValueError("market_provider must be auto, goldapi, or twelvedata.")
        if provider == "goldapi" and not self.gold_api_key.strip():
            raise ValueError("GOLD_API_KEY is required when LIVE_MARKET_PROVIDER=goldapi.")
        if provider == "twelvedata" and not self.twelvedata_api_key.strip():
            raise ValueError(
                "TWELVEDATA_API_KEY is required when LIVE_MARKET_PROVIDER=twelvedata."
            )
        if self.gold_api_key.strip() and self.gold_api_symbol.strip().upper() != "XAG":
            raise ValueError("Gold API Silver source requires GOLD_API_SYMBOL=XAG.")
        if not 1 <= int(self.collection_delay_minutes) <= 30:
            raise ValueError("collection_delay_minutes must be between 1 and 30.")

        bv_mode = self.bullionvault_access_mode.strip().lower()
        if bv_mode not in {"auto", "authenticated", "public"}:
            raise ValueError(
                "bullionvault_access_mode must be auto, authenticated, or public."
            )
        has_bv_user = bool(self.bullionvault_username.strip())
        has_bv_password = bool(self.bullionvault_password)
        if has_bv_user != has_bv_password:
            raise ValueError(
                "BULLIONVAULT_USERNAME and BULLIONVAULT_PASSWORD must be configured together."
            )
        if bv_mode == "authenticated" and not (has_bv_user and has_bv_password):
            raise ValueError(
                "BullionVault authenticated mode requires username and password."
            )
        if self.bullionvault_security_id.strip().upper() not in {
            "AGXLN",
            "AGXZU",
            "AGXTR",
            "AGXSG",
        }:
            raise ValueError(
                "BullionVault Silver security must be AGXLN, AGXZU, AGXTR, or AGXSG."
            )
        if self.bullionvault_currency.strip().upper() != "USD":
            raise ValueError("BullionVault integration currently requires USD quotes.")
        if not 1 <= int(self.bullionvault_market_width) <= 20:
            raise ValueError("BULLIONVAULT_MARKET_WIDTH must be between 1 and 20.")
        if (
            not math.isfinite(float(self.bullionvault_minimum_quantity_kg))
            or float(self.bullionvault_minimum_quantity_kg) <= 0
        ):
            raise ValueError("BULLIONVAULT_MINIMUM_QUANTITY_KG must be finite and positive.")

    @classmethod
    def from_environment(cls, repository_root: Path | None = None) -> "LiveSettings":
        allowed = tuple(
            value.strip()
            for value in os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",")
            if value.strip()
        )
        return cls(
            repository_root=(repository_root or Path(os.getenv("METALPREDICTOR_ROOT", "."))).resolve(),
            database_path=Path(
                os.getenv("LIVE_DB_PATH", "runtime/live_predictions.sqlite3")
            ).expanduser(),
            admin_token=os.getenv("LIVE_ADMIN_TOKEN", "").strip(),
            market_provider=os.getenv("LIVE_MARKET_PROVIDER", "auto").strip().lower() or "auto",
            gold_api_key=os.getenv("GOLD_API_KEY", "").strip(),
            gold_api_symbol=os.getenv("GOLD_API_SYMBOL", "XAG").strip().upper() or "XAG",
            twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY", "").strip(),
            twelvedata_symbol=os.getenv("TWELVEDATA_SYMBOL", "XAG/USD").strip() or "XAG/USD",
            auto_collect=_environment_bool("LIVE_AUTO_COLLECT", False),
            collection_delay_minutes=int(os.getenv("LIVE_COLLECTION_DELAY_MINUTES", "5")),
            bullionvault_username=os.getenv("BULLIONVAULT_USERNAME", "").strip(),
            bullionvault_password=os.getenv("BULLIONVAULT_PASSWORD", ""),
            bullionvault_access_mode=(
                os.getenv("BULLIONVAULT_ACCESS_MODE", "auto").strip().lower() or "auto"
            ),
            bullionvault_security_id=(
                os.getenv("BULLIONVAULT_SECURITY_ID", "AGXLN").strip().upper() or "AGXLN"
            ),
            bullionvault_currency=(
                os.getenv("BULLIONVAULT_CURRENCY", "USD").strip().upper() or "USD"
            ),
            bullionvault_market_width=int(os.getenv("BULLIONVAULT_MARKET_WIDTH", "5")),
            bullionvault_minimum_quantity_kg=float(
                os.getenv("BULLIONVAULT_MINIMUM_QUANTITY_KG", "0.001")
            ),
            bullionvault_public_fallback=_environment_bool(
                "BULLIONVAULT_PUBLIC_FALLBACK", True
            ),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(),
            telegram_allowed_chat_ids=allowed,
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
        )

    @property
    def resolved_market_provider(self) -> str | None:
        provider = self.market_provider.strip().lower()
        if provider == "goldapi":
            return "GoldAPI" if self.gold_api_key.strip() else None
        if provider == "twelvedata":
            return "TwelveData" if self.twelvedata_api_key.strip() else None
        if self.gold_api_key.strip():
            return "GoldAPI"
        if self.twelvedata_api_key.strip():
            return "TwelveData"
        return None

    @property
    def market_source_enabled(self) -> bool:
        return self.resolved_market_provider is not None

    @property
    def market_source_symbol(self) -> str | None:
        provider = self.resolved_market_provider
        if provider == "GoldAPI":
            return self.gold_api_symbol.strip().upper() or "XAG"
        if provider == "TwelveData":
            return self.twelvedata_symbol.strip() or "XAG/USD"
        return None

    @property
    def market_source_mode(self) -> str | None:
        provider = self.resolved_market_provider
        if provider == "GoldAPI":
            return "PROVIDER_H1_OHLC_LATEST_ONLY"
        if provider == "TwelveData":
            return "M1_TO_H1_CONSERVATIVE_AGGREGATION"
        return None

    @property
    def auto_collection_enabled(self) -> bool:
        return bool(self.auto_collect and self.market_source_enabled)

    @property
    def bullionvault_authenticated_configured(self) -> bool:
        return bool(self.bullionvault_username.strip() and self.bullionvault_password)

    @property
    def bullionvault_resolved_access_mode(self) -> str:
        mode = self.bullionvault_access_mode.strip().lower()
        if mode == "authenticated":
            return "authenticated"
        if mode == "public":
            return "public"
        return "authenticated" if self.bullionvault_authenticated_configured else "public"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_allowed_chat_ids)

    @property
    def telegram_webhook_enabled(self) -> bool:
        return bool(
            self.telegram_enabled
            and self.telegram_webhook_secret
            and self.public_base_url
        )
