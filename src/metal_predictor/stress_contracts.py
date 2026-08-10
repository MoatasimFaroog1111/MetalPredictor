from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HistoricalStressConfig:
    target_name: str = "target_log_return_1h"
    timestamp_name: str = "timestamp_utc"
    target_timestamp_name: str = "target_timestamp_utc"
    current_close_name: str = "close_usd_per_kg"
    target_close_name: str = "target_close_usd_per_kg"
    current_quality_name: str = "quality_flag"
    target_quality_name: str = "target_quality_flag"
    strict_quality_value: str = "OK"
    model_names: tuple[str, ...] = ("ridge_alpha_10", "ridge_alpha_100")
    block_sizes_rows: tuple[int, ...] = (24, 120)
    bootstrap_resamples: int = 5000
    random_state: int = 42


@dataclass(frozen=True)
class StressFoldMetric:
    protocol: str
    year: int
    model: str
    train_rows: int
    validation_rows: int
    train_first_timestamp_utc: str
    train_last_target_timestamp_utc: str
    validation_first_timestamp_utc: str
    validation_last_timestamp_utc: str
    mae_return: float
    rmse_return: float
    r2_return: float
    directional_accuracy: float
    balanced_directional_accuracy: float
    strategy_mean_log_return: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
