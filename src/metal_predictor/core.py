from __future__ import annotations
from pathlib import Path
from typing import Protocol
import pandas as pd

class DataLoader(Protocol):
    def load(self, path: Path) -> pd.DataFrame: ...

class DataValidator(Protocol):
    def validate(self, frame: pd.DataFrame) -> None: ...

class FeatureComponent(Protocol):
    @property
    def feature_names(self) -> tuple[str, ...]: ...
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...

class TargetBuilder(Protocol):
    @property
    def target_names(self) -> tuple[str, ...]: ...
    def build(self, frame: pd.DataFrame) -> pd.DataFrame: ...

class DatasetSplitter(Protocol):
    def split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]: ...

class LeakageGuard(Protocol):
    def validate(self, full_frame: pd.DataFrame, splits: dict[str, pd.DataFrame],
                 feature_names: tuple[str, ...], target_names: tuple[str, ...]) -> None: ...

class ArtifactWriter(Protocol):
    def write(self, splits: dict[str, pd.DataFrame], feature_names: tuple[str, ...],
              target_names: tuple[str, ...], output_dir: Path) -> None: ...

from dataclasses import dataclass, field

@dataclass(frozen=True)
class ColumnConfig:
    timestamp: str = "timestamp_utc"
    open: str = "open_usd_per_kg"
    high: str = "high_usd_per_kg"
    low: str = "low_usd_per_kg"
    close: str = "close_usd_per_kg"
    quality: str = "quality_flag"

    @property
    def required(self) -> tuple[str, ...]:
        return (self.timestamp, self.open, self.high, self.low, self.close)

@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168)
    volatility_windows: tuple[int, ...] = (6, 12, 24, 72, 168)
    trend_windows: tuple[int, ...] = (12, 24, 72, 168)
    rsi_window: int = 14
    atr_window: int = 14

@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_hours: int = 1

    def __post_init__(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be positive.")
        if self.purge_hours < 1:
            raise ValueError("purge_hours must be >= 1 for a future target.")

@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
    output_dir: Path = Path("data/processed")
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    target_horizon_hours: int = 1
    strict_quality_only: bool = False

    def __post_init__(self) -> None:
        if self.target_horizon_hours != 1:
            raise ValueError("Current target component supports exactly a 1-hour horizon.")
