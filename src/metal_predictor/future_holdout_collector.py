from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from metal_predictor.append_only_ledger import CsvHashChainLedger
from metal_predictor.frozen_ridge import FrozenRidgeRegressor
from metal_predictor.future_features import SilverFeatureAssembler
from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import (
    DownloadWindow,
    GenericAsciiM1Parser,
    HistDataArchiveDownloader,
    InstrumentSpec,
)


OBSERVATION_COLUMNS = (
    "timestamp_utc",
    "holdout_role",
    "open_usd_per_oz",
    "high_usd_per_oz",
    "low_usd_per_oz",
    "close_usd_per_oz",
    "open_usd_per_kg",
    "high_usd_per_kg",
    "low_usd_per_kg",
    "close_usd_per_kg",
    "minute_count",
    "quality_flag",
    "source_provider",
    "source_symbol",
    "market_type",
)

PREDICTION_COLUMNS = (
    "feature_timestamp_utc",
    "decision_time_utc",
    "primary_model_name",
    "primary_prediction_log_return_1h",
    "primary_model_hash",
    "benchmark_model_name",
    "benchmark_prediction_log_return_1h",
    "benchmark_model_hash",
    "feature_vector_sha256",
)


@dataclass(frozen=True)
class CollectionReport:
    collected_at_utc: str
    source_batch_rows: int
    source_overlap_rows_verified: int
    observations_appended: int
    observation_ledger_rows: int
    predictions_appended: int
    prediction_ledger_rows: int
    latest_observation_timestamp_utc: str | None
    latest_prediction_timestamp_utc: str | None
    performance_metrics_computed: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FutureHoldoutCollector:
    """Append-only future XAG collector and frozen-model prediction materializer.

    The collector intentionally computes NO performance metrics. HistData is a batch
    historical source, so predictions may be materialized after their outcomes exist;
    validity comes from the fact that model parameters, feature code and evaluation
    rules were frozen before the holdout data existed. This is a frozen future
    holdout, not evidence of live execution latency or achievable fills.
    """

    def __init__(self, frozen_root: Path, ledger_root: Path) -> None:
        self._frozen_root = frozen_root
        self._ledger_root = ledger_root
        self._manifest = json.loads(
            (frozen_root / "forward_holdout/freeze_manifest.json").read_text(encoding="utf-8")
        )
        self._observations = CsvHashChainLedger(
            ledger_root / "forward_holdout/observations.csv",
            OBSERVATION_COLUMNS,
            "timestamp_utc",
        )
        self._predictions = CsvHashChainLedger(
            ledger_root / "forward_holdout/predictions.csv",
            PREDICTION_COLUMNS,
            "feature_timestamp_utc",
        )
        self._primary = FrozenRidgeRegressor.from_path(
            frozen_root / self._manifest["models"]["primary"]["path"]
        )
        self._benchmark = FrozenRidgeRegressor.from_path(
            frozen_root / self._manifest["models"]["benchmark"]["path"]
        )
        self._assembler = SilverFeatureAssembler()
        if self._primary.feature_names != self._benchmark.feature_names:
            raise ValueError("Primary and benchmark frozen feature lists differ.")
        if tuple(self._assembler.feature_names) != self._primary.feature_names:
            raise ValueError("Frozen feature graph no longer matches exported model feature order.")

    def collect(self, now_utc: pd.Timestamp | None = None) -> CollectionReport:
        now = pd.Timestamp(now_utc or datetime.now(timezone.utc)).tz_convert("UTC")
        context_start = pd.Timestamp(self._manifest["context_start_utc"])
        holdout_start = pd.Timestamp(self._manifest["holdout_first_bar_start_utc"])
        if now <= context_start + pd.Timedelta(hours=2):
            return self._report(now, 0, 0, 0, 0)

        existing_obs = self._observations.read_verified()
        if len(existing_obs):
            last_stored = pd.Timestamp(existing_obs["timestamp_utc"].iloc[-1])
            source_start = max(context_start, last_stored - pd.Timedelta(days=7))
        else:
            source_start = context_start
        source_end = now.floor("h") - pd.Timedelta(hours=1)
        if source_end <= source_start:
            return self._report(now, 0, 0, 0, 0)

        source = self._download_hourly(source_start, source_end)
        source = source.loc[
            pd.to_datetime(source["timestamp_utc"], utc=True).ge(context_start)
        ].copy()
        source["holdout_role"] = np.where(
            pd.to_datetime(source["timestamp_utc"], utc=True).ge(holdout_start),
            "HOLDOUT",
            "CONTEXT",
        )
        source_rows = self._observation_view(source)
        overlap_count, append_rows = self._verify_overlap_and_select_new(
            existing_obs, source_rows
        )
        observation_result = self._observations.append(append_rows)

        prediction_rows = self._new_predictions()
        prediction_result = self._predictions.append(prediction_rows)
        report = self._report(
            now,
            len(source_rows),
            overlap_count,
            observation_result.appended_rows,
            prediction_result.appended_rows,
        )
        state_path = self._ledger_root / "forward_holdout/collection_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        return report

    def _download_hourly(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        instrument = InstrumentSpec(asset="XAG", pair="xagusd", source_symbol="XAGUSD")
        window = DownloadWindow(start_utc=start, end_utc=end)
        with TemporaryDirectory(prefix="metalpredictor-histdata-") as directory:
            archives = HistDataArchiveDownloader().download(
                instrument, window, Path(directory)
            )
            minutes = GenericAsciiM1Parser().parse(archives)
            hourly, _ = ConservativeH1Aggregator().aggregate(
                minutes, instrument, window
            )
        return hourly

    @staticmethod
    def _observation_view(source: pd.DataFrame) -> pd.DataFrame:
        missing = set(OBSERVATION_COLUMNS).difference(source.columns)
        if missing:
            raise ValueError(f"Aggregated XAG data missing holdout columns: {sorted(missing)}")
        return source.loc[:, OBSERVATION_COLUMNS].copy().sort_values("timestamp_utc")

    def _verify_overlap_and_select_new(
        self,
        existing: pd.DataFrame,
        source: pd.DataFrame,
    ) -> tuple[int, pd.DataFrame]:
        if existing.empty:
            return 0, source.reset_index(drop=True)
        existing_indexed = existing.set_index("timestamp_utc")
        source_copy = source.copy()
        source_copy["timestamp_utc"] = pd.to_datetime(
            source_copy["timestamp_utc"], utc=True
        ).map(lambda value: value.isoformat())
        source_indexed = source_copy.set_index("timestamp_utc")
        overlap = existing_indexed.index.intersection(source_indexed.index)
        compare_numeric = (
            "open_usd_per_oz", "high_usd_per_oz", "low_usd_per_oz", "close_usd_per_oz",
            "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg",
            "minute_count",
        )
        compare_text = (
            "holdout_role", "quality_flag", "source_provider", "source_symbol", "market_type",
        )
        for timestamp in overlap:
            left = existing_indexed.loc[timestamp]
            right = source_indexed.loc[timestamp]
            for column in compare_numeric:
                a = float(left[column])
                b = float(right[column])
                if not np.isclose(a, b, rtol=1e-12, atol=1e-10):
                    raise ValueError(
                        f"SOURCE_REVISION_CONFLICT {timestamp} {column}: ledger={a} source={b}"
                    )
            for column in compare_text:
                if str(left[column]) != str(right[column]):
                    raise ValueError(
                        f"SOURCE_REVISION_CONFLICT {timestamp} {column}: "
                        f"ledger={left[column]!r} source={right[column]!r}"
                    )
        last_existing = pd.Timestamp(existing["timestamp_utc"].iloc[-1])
        newer = source.loc[
            pd.to_datetime(source["timestamp_utc"], utc=True).gt(last_existing)
        ].copy()
        return len(overlap), newer.reset_index(drop=True)

    def _new_predictions(self) -> pd.DataFrame:
        obs = self._observations.read_verified()
        if obs.empty:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        pred = self._predictions.read_verified()
        last_prediction = (
            pd.Timestamp(pred["feature_timestamp_utc"].iloc[-1])
            if len(pred) else None
        )
        eligible = obs.loc[obs["holdout_role"].eq("HOLDOUT")].copy()
        if last_prediction is not None:
            eligible = eligible.loc[
                pd.to_datetime(eligible["timestamp_utc"], utc=True).gt(last_prediction)
            ].copy()
        if eligible.empty:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)

        historical = pd.read_parquet(
            self._frozen_root / self._manifest["historical_source_dataset_path"]
        )
        observation_hourly = self._ledger_to_hourly(obs)
        combined = pd.concat([historical, observation_hourly], ignore_index=True, sort=False)
        combined["timestamp_utc"] = pd.to_datetime(
            combined["timestamp_utc"], utc=True, errors="raise"
        )
        combined = (
            combined.sort_values("timestamp_utc")
            .drop_duplicates("timestamp_utc", keep="last")
            .reset_index(drop=True)
        )
        featured = self._assembler.transform(combined)
        featured = featured.set_index("timestamp_utc")
        wanted = pd.DatetimeIndex(pd.to_datetime(eligible["timestamp_utc"], utc=True))
        rows = featured.reindex(wanted)
        if rows.index.has_duplicates or len(rows) != len(wanted):
            raise ValueError("Future feature reindex failed.")
        primary_prediction = self._primary.predict(rows)
        benchmark_prediction = self._benchmark.predict(rows)
        output: list[dict[str, object]] = []
        for index, timestamp in enumerate(wanted):
            feature_hash = self._feature_vector_hash(rows.iloc[index])
            output.append({
                "feature_timestamp_utc": timestamp.isoformat(),
                "decision_time_utc": (timestamp + pd.Timedelta(hours=1)).isoformat(),
                "primary_model_name": self._primary.model_name,
                "primary_prediction_log_return_1h": primary_prediction[index],
                "primary_model_hash": self._primary.model_payload_sha256,
                "benchmark_model_name": self._benchmark.model_name,
                "benchmark_prediction_log_return_1h": benchmark_prediction[index],
                "benchmark_model_hash": self._benchmark.model_payload_sha256,
                "feature_vector_sha256": feature_hash,
            })
        return pd.DataFrame(output, columns=PREDICTION_COLUMNS)

    @staticmethod
    def _ledger_to_hourly(obs: pd.DataFrame) -> pd.DataFrame:
        columns = (
            "timestamp_utc", "open_usd_per_kg", "high_usd_per_kg",
            "low_usd_per_kg", "close_usd_per_kg", "minute_count", "quality_flag",
        )
        out = obs.loc[:, columns].copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        for column in (
            "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"
        ):
            out[column] = pd.to_numeric(out[column], errors="raise")
        out["minute_count"] = pd.to_numeric(out["minute_count"], errors="raise")
        return out

    def _feature_vector_hash(self, row: pd.Series) -> str:
        values = []
        for feature in self._primary.feature_names:
            value = row[feature]
            values.append(None if pd.isna(value) else format(float(value), ".17g"))
        canonical = json.dumps(
            {"features": list(self._primary.feature_names), "values": values},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _report(
        self,
        now: pd.Timestamp,
        source_batch_rows: int,
        overlap_rows: int,
        observations_appended: int,
        predictions_appended: int,
    ) -> CollectionReport:
        obs = self._observations.read_verified()
        pred = self._predictions.read_verified()
        return CollectionReport(
            collected_at_utc=now.isoformat(),
            source_batch_rows=source_batch_rows,
            source_overlap_rows_verified=overlap_rows,
            observations_appended=observations_appended,
            observation_ledger_rows=len(obs),
            predictions_appended=predictions_appended,
            prediction_ledger_rows=len(pred),
            latest_observation_timestamp_utc=(
                str(obs["timestamp_utc"].iloc[-1]) if len(obs) else None
            ),
            latest_prediction_timestamp_utc=(
                str(pred["feature_timestamp_utc"].iloc[-1]) if len(pred) else None
            ),
        )
