from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.frozen_ridge import FrozenRidgeRegressor
from metal_predictor.future_features import SilverFeatureAssembler
from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.precious_metals.confirmation import CANDIDATE_FEATURES, CANDIDATE_ID
from metal_predictor.shadow62.contracts import ShadowForecastSnapshot
from metal_predictor.shadow62.features import ConfirmedPreciousMetalsShadowFeatures


class Shadow62InferenceEngine:
    """Research-only parallel inference for the locked 52+10 candidate.

    The production LivePredictionEngine is not imported or mutated. This engine loads
    independent frozen payloads, reconstructs the same causal Silver 52-feature graph,
    adds only the ten locked XPT/XPD features, and emits an auditable shadow snapshot.
    """

    def __init__(self, repository_root: Path) -> None:
        self._root = Path(repository_root).resolve()
        manifest_path = self._root / "shadow_holdout/freeze_manifest.json"
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self._manifest.get("candidate_id") != CANDIDATE_ID:
            raise ValueError("Shadow freeze manifest candidate_id does not match locked candidate.")
        if self._manifest.get("candidate_features") != list(CANDIDATE_FEATURES):
            raise ValueError("Shadow freeze manifest candidate feature list changed.")

        historical_path = self._root / self._manifest["historical_source_dataset_path"]
        self._historical = pd.read_parquet(historical_path).copy(deep=True)
        self._historical["timestamp_utc"] = pd.to_datetime(
            self._historical["timestamp_utc"], utc=True, errors="raise"
        )
        self._historical = self._historical.sort_values("timestamp_utc").reset_index(drop=True)
        self._historical_last = pd.Timestamp(self._historical["timestamp_utc"].iloc[-1])

        baseline_path = self._root / self._manifest["models"]["baseline"]["path"]
        candidate_path = self._root / self._manifest["models"]["candidate"]["path"]
        self._baseline = FrozenRidgeRegressor.from_path(baseline_path)
        self._candidate = FrozenRidgeRegressor.from_path(candidate_path)
        self._silver = SilverFeatureAssembler()
        self._precious = ConfirmedPreciousMetalsShadowFeatures()

        if tuple(self._silver.feature_names) != self._baseline.feature_names:
            raise ValueError("Shadow baseline does not match the frozen Silver 52-feature graph.")
        expected_candidate = self._baseline.feature_names + CANDIDATE_FEATURES
        if self._candidate.feature_names != expected_candidate:
            raise ValueError("Shadow candidate payload is not exactly Frozen52 + locked10.")

    @property
    def candidate_model_sha256(self) -> str:
        return self._candidate.model_payload_sha256

    @property
    def baseline_model_sha256(self) -> str:
        return self._baseline.model_payload_sha256

    def predict(
        self,
        live_bars: list[HourlySilverBar],
        platinum_frame: pd.DataFrame,
        palladium_frame: pd.DataFrame,
        *,
        materialized_at_utc: datetime | None = None,
    ) -> ShadowForecastSnapshot:
        if not live_bars:
            raise ValueError("Shadow inference requires at least one live Silver bar.")
        ordered = sorted(live_bars, key=lambda item: item.timestamp_utc)
        live = self._bars_frame(ordered)
        newest = pd.Timestamp(live["timestamp_utc"].iloc[-1])
        if newest <= self._historical_last:
            raise ValueError("Shadow feature timestamp must be newer than frozen historical context.")
        if live["timestamp_utc"].le(self._historical_last).any():
            raise ValueError("Shadow live Silver repository overlaps frozen historical context.")

        combined = pd.concat([self._historical, live], ignore_index=True, sort=False)
        combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
        silver_featured = self._silver.transform(combined)
        row = silver_featured.loc[silver_featured["timestamp_utc"].eq(newest)].copy(deep=True)
        if len(row) != 1:
            raise ValueError("Shadow engine could not resolve exactly one newest Silver feature row.")
        row = self._precious.transform(row, platinum_frame, palladium_frame)
        self._validate_features(row)

        baseline_return = float(self._baseline.predict(row)[0])
        candidate_return = float(self._candidate.predict(row)[0])
        close = float(row["close_usd_per_kg"].iloc[0])
        materialized = (materialized_at_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

        return ShadowForecastSnapshot(
            feature_timestamp_utc=newest.to_pydatetime(),
            materialized_at_utc=materialized,
            reference_close_usd_per_kg=close,
            baseline_model=self._baseline.model_name,
            baseline_model_sha256=self._baseline.model_payload_sha256,
            baseline_log_return_1h=baseline_return,
            baseline_predicted_price_usd_per_kg=close * math.exp(baseline_return),
            candidate_id=CANDIDATE_ID,
            candidate_model=self._candidate.model_name,
            candidate_model_sha256=self._candidate.model_payload_sha256,
            candidate_log_return_1h=candidate_return,
            candidate_predicted_price_usd_per_kg=close * math.exp(candidate_return),
            xpt_exact_current=self._precious.exact_current_available(platinum_frame, newest),
            xpd_exact_current=self._precious.exact_current_available(palladium_frame, newest),
            auxiliary_provider="Dukascopy Public Historical Feed / H1 Bid",
        )

    def status(self) -> dict[str, object]:
        return {
            "candidate_id": CANDIDATE_ID,
            "candidate_feature_count": len(CANDIDATE_FEATURES),
            "total_model_feature_count": len(self._candidate.feature_names),
            "baseline_model": self._baseline.model_name,
            "baseline_model_sha256": self._baseline.model_payload_sha256,
            "candidate_model": self._candidate.model_name,
            "candidate_model_sha256": self._candidate.model_payload_sha256,
            "historical_context_last_timestamp_utc": self._historical_last.isoformat(),
            "edge_status": "NOT_PROVEN",
            "research_only": True,
            "live_model_mutated": False,
            "frozen_52_feature_graph_mutated": False,
            "buy_sell_enabled": False,
            "execution_enabled": False,
        }

    def _validate_features(self, row: pd.DataFrame) -> None:
        missing = set(self._candidate.feature_names).difference(row.columns)
        if missing:
            raise ValueError(f"Shadow candidate input missing features: {sorted(missing)}")
        values = row.loc[:, self._candidate.feature_names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Shadow candidate features contain infinite values.")

    @staticmethod
    def _bars_frame(bars: list[HourlySilverBar]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        previous: datetime | None = None
        for bar in bars:
            if bar.timestamp_utc.tzinfo is None:
                raise ValueError("Shadow Silver bar timestamps must be timezone-aware.")
            timestamp = bar.timestamp_utc.astimezone(timezone.utc)
            if previous is not None and timestamp <= previous:
                raise ValueError("Shadow Silver bars must be unique and chronological.")
            previous = timestamp
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "open_usd_per_kg": float(bar.open_usd_per_kg),
                    "high_usd_per_kg": float(bar.high_usd_per_kg),
                    "low_usd_per_kg": float(bar.low_usd_per_kg),
                    "close_usd_per_kg": float(bar.close_usd_per_kg),
                    "minute_count": int(bar.minute_count),
                    "quality_flag": str(bar.quality_flag),
                }
            )
        frame = pd.DataFrame(rows)
        prices = frame[
            ["open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"]
        ].to_numpy(float)
        if not np.isfinite(prices).all():
            raise ValueError("Shadow Silver bars contain non-finite prices.")
        return frame
