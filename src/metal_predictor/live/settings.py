from __future__ import annotations

from dataclasses import dataclass
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
    twelvedata_api_key: str = ""
    twelvedata_symbol: str = "XAG/USD"
    auto_collect: bool = False
    collection_delay_minutes: int = 5
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = ""
    telegram_allowed_chat_ids: tuple[str, ...] = ()
    public_base_url: str = ""

    def __post_init__(self) -> None:
        if not 1 <= int(self.collection_delay_minutes) <= 30:
            raise ValueError("collection_delay_minutes must be between 1 and 30.")

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
            twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY", "").strip(),
            twelvedata_symbol=os.getenv("TWELVEDATA_SYMBOL", "XAG/USD").strip() or "XAG/USD",
            auto_collect=_environment_bool("LIVE_AUTO_COLLECT", False),
            collection_delay_minutes=int(os.getenv("LIVE_COLLECTION_DELAY_MINUTES", "5")),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(),
            telegram_allowed_chat_ids=allowed,
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
        )

    @property
    def market_source_enabled(self) -> bool:
        return bool(self.twelvedata_api_key)

    @property
    def auto_collection_enabled(self) -> bool:
        return bool(self.auto_collect and self.market_source_enabled)

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
