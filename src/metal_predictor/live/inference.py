from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.frozen_ridge import FrozenRidgeRegressor
from metal_predictor.future_features import SilverFeatureAssembler
from metal_predictor.live.contracts import ForecastSnapshot, HourlySilverBar


class LivePredictionEngine:
    """Causal operational inference using frozen model payloads only.

    No fitting, tuning, threshold optimization, or performance measurement occurs here.
    Ridge(alpha=100) stays the operational baseline; Ridge(alpha=10) is exposed only
    as the predeclared research challenger from Stage 7.
    """

    def __init__(self, repository_root: Path) -> None:
        self._root = Path(repository_root).resolve()
        manifest_path = self._root / "forward_holdout/freeze_manifest.json"
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._historical_path = self._root / self._manifest["historical_source_dataset_path"]
        self._historical = pd.read_parquet(self._historical_path).copy(deep=True)
        self._historical["timestamp_utc"] = pd.to_datetime(
            self._historical["timestamp_utc"], utc=True, errors="raise"
        )
        self._historical = self._historical.sort_values("timestamp_utc").reset_index(drop=True)
        self._historical_last = pd.Timestamp(self._historical["timestamp_utc"].iloc[-1])

        baseline_path = self._root / self._manifest["models"]["benchmark"]["path"]
        challenger_path = self._root / self._manifest["models"]["primary"]["path"]
        self._baseline = FrozenRidgeRegressor.from_path(baseline_path)
        self._challenger = FrozenRidgeRegressor.from_path(challenger_path)
        self._assembler = SilverFeatureAssembler()
        if self._baseline.feature_names != self._challenger.feature_names:
            raise ValueError("Frozen baseline and challenger feature lists differ.")
        if tuple(self._assembler.feature_names) != self._baseline.feature_names:
            raise ValueError("Live feature graph does not match frozen model feature order.")

    @property
    def baseline_model_name(self) -> str:
        return self._baseline.model_name

    @property
    def challenger_model_name(self) -> str:
        return self._challenger.model_name

    @property
    def feature_count(self) -> int:
        return len(self._baseline.feature_names)

    @property
    def historical_last_timestamp_utc(self) -> str:
        return self._historical_last.isoformat()

    @property
    def historical_last_datetime_utc(self) -> datetime:
        return self._historical_last.to_pydatetime()

    def predict(self, live_bars: list[HourlySilverBar]) -> ForecastSnapshot:
        if not live_bars:
            raise ValueError("At least one live bar is required for live prediction.")
        ordered_bars = sorted(live_bars, key=lambda item: item.timestamp_utc)
        live = self._bars_frame(ordered_bars)
        newest = pd.Timestamp(live["timestamp_utc"].iloc[-1])
        if newest <= self._historical_last:
            raise ValueError(
                "Latest live timestamp must be newer than the frozen historical context."
            )

        overlap = live.loc[live["timestamp_utc"].le(self._historical_last)]
        if len(overlap):
            raise ValueError(
                "Live repository contains timestamps inside frozen historical context; "
                "overlap must be resolved before inference."
            )

        combined = pd.concat([self._historical, live], ignore_index=True, sort=False)
        combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
        featured = self._assembler.transform(combined)
        row = featured.loc[featured["timestamp_utc"].eq(newest)]
        if len(row) != 1:
            raise ValueError("Could not resolve exactly one feature row for latest live hour.")
        self._require_complete_features(row, newest)

        baseline_return = float(self._baseline.predict(row)[0])
        challenger_return = float(self._challenger.predict(row)[0])
        close = float(row["close_usd_per_kg"].iloc[0])
        baseline_price = close * math.exp(baseline_return)
        challenger_price = close * math.exp(challenger_return)
        latest_bar = ordered_bars[-1]

        return ForecastSnapshot(
            feature_timestamp_utc=newest.to_pydatetime(),
            decision_time_utc=(newest + pd.Timedelta(hours=1)).to_pydatetime(),
            current_price_usd_per_kg=close,
            baseline_model=self._baseline.model_name,
            baseline_log_return_1h=baseline_return,
            baseline_predicted_price_usd_per_kg=baseline_price,
            baseline_direction=self._direction(baseline_return),
            challenger_model=self._challenger.model_name,
            challenger_log_return_1h=challenger_return,
            challenger_predicted_price_usd_per_kg=challenger_price,
            challenger_direction=self._direction(challenger_return),
            data_quality=latest_bar.quality_flag,
            source_provider=latest_bar.source_provider,
            source_compatible_with_training=(
                latest_bar.source_provider == "HistData"
                and latest_bar.market_type == "spot_bid"
            ),
        )

    def model_status(self) -> dict[str, object]:
        return {
            "baseline_model": self._baseline.model_name,
            "baseline_model_sha256": self._baseline.model_payload_sha256,
            "challenger_model": self._challenger.model_name,
            "challenger_model_sha256": self._challenger.model_payload_sha256,
            "feature_count": self.feature_count,
            "target": "next exact-hour XAG/USD log return",
            "historical_context_last_timestamp_utc": self.historical_last_timestamp_utc,
            "edge_status": "NOT_PROVEN",
            "research_only": True,
            "buy_sell_enabled": False,
        }

    def _require_complete_features(
        self,
        row: pd.DataFrame,
        newest: pd.Timestamp,
    ) -> None:
        feature_frame = row.loc[:, self._baseline.feature_names]
        numeric = feature_frame.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if np.isfinite(values).all():
            return
        missing = [
            name
            for name in self._baseline.feature_names
            if not np.isfinite(float(numeric[name].iloc[0]))
        ]
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            f"LIVE_FEATURES_INCOMPLETE {newest.isoformat()} missing={preview}{suffix}"
        )

    @staticmethod
    def _direction(value: float) -> str:
        if not np.isfinite(value):
            raise ValueError("Prediction must be finite.")
        if value > 0:
            return "UP"
        if value < 0:
            return "DOWN"
        return "FLAT"

    @staticmethod
    def _bars_frame(bars: list[HourlySilverBar]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        previous: datetime | None = None
        for bar in bars:
            if bar.timestamp_utc.tzinfo is None:
                raise ValueError("Live bar timestamps must be timezone-aware.")
            ts = bar.timestamp_utc.astimezone(timezone.utc)
            if previous is not None and ts <= previous:
                raise ValueError("Live bars must be unique and chronological.")
            previous = ts
            rows.append(
                {
                    "timestamp_utc": ts,
                    "open_usd_per_kg": float(bar.open_usd_per_kg),
                    "high_usd_per_kg": float(bar.high_usd_per_kg),
                    "low_usd_per_kg": float(bar.low_usd_per_kg),
                    "close_usd_per_kg": float(bar.close_usd_per_kg),
                    "minute_count": int(bar.minute_count),
                    "quality_flag": str(bar.quality_flag),
                }
            )
        frame = pd.DataFrame(rows)
        values = frame[[
            "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"
        ]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Live bar frame contains non-finite prices.")
        return frame


class LiveForecastOrchestrator:
    """Coordinates storage, frozen inference, and optional notification."""

    def __init__(self, repository, engine: LivePredictionEngine, notifier=None) -> None:
        self._repository = repository
        self._engine = engine
        self._notifier = notifier

    def ingest_bar(self, bar: HourlySilverBar) -> bool:
        return self._repository.put_bar(bar)

    def materialize_latest_forecast(self) -> tuple[ForecastSnapshot, bool]:
        bars = self._repository.recent_bars(limit=5000)
        if not bars:
            raise ValueError("No live bars are available for forecast materialization.")
        snapshot = self._engine.predict(bars)
        forecast_created = self._repository.put_forecast(snapshot)
        if forecast_created and self._notifier is not None:
            self._notifier.publish_forecast(snapshot)
        return snapshot, forecast_created

    def ingest_and_forecast(self, bar: HourlySilverBar) -> tuple[ForecastSnapshot, bool, bool]:
        bar_created = self.ingest_bar(bar)
        snapshot, forecast_created = self.materialize_latest_forecast()
        return snapshot, bar_created, forecast_created

    def latest(self) -> ForecastSnapshot | None:
        return self._repository.latest_forecast()

    @staticmethod
    def previous_completed_hour(now_utc: datetime | None = None) -> datetime:
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        return current_hour - timedelta(hours=1)
