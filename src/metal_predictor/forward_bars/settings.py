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
class ForwardBarSettings:
    enabled: bool = False
    database_path: Path = Path("runtime/bullionvault_forward_bars.sqlite3")
    materialization_interval_seconds: int = 60
    close_delay_seconds: int = 120
    max_buckets_per_cycle: int = 512

    def __post_init__(self) -> None:
        if not 30 <= int(self.materialization_interval_seconds) <= 3600:
            raise ValueError("BULLIONVAULT_FORWARD_BARS_MATERIALIZATION_INTERVAL_SECONDS must be between 30 and 3600.")
        if not 30 <= int(self.close_delay_seconds) <= 3600:
            raise ValueError("BULLIONVAULT_FORWARD_BARS_CLOSE_DELAY_SECONDS must be between 30 and 3600.")
        if not 1 <= int(self.max_buckets_per_cycle) <= 10_000:
            raise ValueError("BULLIONVAULT_FORWARD_BARS_MAX_BUCKETS_PER_CYCLE must be between 1 and 10000.")

    @classmethod
    def from_environment(cls) -> "ForwardBarSettings":
        return cls(
            enabled=_environment_bool("BULLIONVAULT_FORWARD_BARS_ENABLED", False),
            database_path=Path(os.getenv("BULLIONVAULT_FORWARD_BARS_DB_PATH", "runtime/bullionvault_forward_bars.sqlite3")).expanduser(),
            materialization_interval_seconds=int(os.getenv("BULLIONVAULT_FORWARD_BARS_MATERIALIZATION_INTERVAL_SECONDS", "60")),
            close_delay_seconds=int(os.getenv("BULLIONVAULT_FORWARD_BARS_CLOSE_DELAY_SECONDS", "120")),
            max_buckets_per_cycle=int(os.getenv("BULLIONVAULT_FORWARD_BARS_MAX_BUCKETS_PER_CYCLE", "512")),
        )
