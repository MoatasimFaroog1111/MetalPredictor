from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.frozen_ridge import FrozenRidgeExporter, FrozenRidgeRegressor
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.precious_metals.confirmation import (
    CANDIDATE_FEATURES,
    CANDIDATE_FAMILIES,
    CANDIDATE_ID,
    candidate_fingerprint,
)
from metal_predictor.shadow62.contracts import (
    SHADOW_EARLIEST_FINAL_SCORE_UTC,
    SHADOW_FIRST_FEATURE_BAR_START_UTC,
    SHADOW_FIXED_WINDOW_DAYS,
    SHADOW_FREEZE_ID,
    SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC,
    SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES,
    SHADOW_PROTOCOL_VERSION,
)


ENHANCED_DIR = Path("data/processed_precious_metals")
BASE_DIR = Path("data/processed")
OUTPUT_DIR = Path("shadow_holdout")
SILVER_RAW_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
SILVER_RAW_GIT_BLOB_SHA = "9b95fcc5aa2679208c6b5c44c830ce6b1eaa5829"
CONFIRMED_CANDIDATE_FINGERPRINT = "41b8089018654b8f722bd01b129031a49960c2054c6c6e64f442cef0c65be329"
HISTORICAL_CONFIRMATION_RUN_ID = 31603954945


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training() -> tuple[pd.DataFrame, tuple[str, ...]]:
    manifest = json.loads((BASE_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    base_features = tuple(str(value) for value in manifest["features"])
    candidate_features = base_features + CANDIDATE_FEATURES

    frames: list[pd.DataFrame] = []
    for name in ("train", "validation", "test"):
        path = ENHANCED_DIR / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        frame["target_timestamp_utc"] = pd.to_datetime(
            frame["target_timestamp_utc"], utc=True, errors="raise"
        )
        frames.append(frame)

    training = pd.concat(frames, ignore_index=True)
    training = training.sort_values("timestamp_utc").reset_index(drop=True)
    if training.empty or training["timestamp_utc"].duplicated().any():
        raise ValueError("Shadow62 freeze training rows are empty or duplicated.")
    if not training["target_timestamp_utc"].sub(training["timestamp_utc"]).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Shadow62 freeze requires exact next-hour targets.")
    missing = set(candidate_features).difference(training.columns)
    if missing:
        raise ValueError(f"Shadow62 freeze missing locked features: {sorted(missing)}")
    return training, candidate_features


def _ridge_alpha_100():
    return next(
        spec
        for spec in DefaultModelRegistry(random_state=42).candidates()
        if spec.name == "ridge_alpha_100"
    )


def _equivalence_error(spec, payload, training, feature_names) -> float:
    sklearn_model = spec.factory()
    sklearn_model.fit(training.loc[:, feature_names], training["target_log_return_1h"])
    positions = np.linspace(0, len(training) - 1, num=min(512, len(training)), dtype=int)
    sample = training.iloc[positions]
    expected = np.asarray(sklearn_model.predict(sample.loc[:, feature_names]), dtype=float)
    frozen = FrozenRidgeRegressor(payload).predict(sample.loc[:, feature_names])
    return float(np.max(np.abs(expected - frozen)))


def build() -> dict[str, object]:
    if candidate_fingerprint() != CONFIRMED_CANDIDATE_FINGERPRINT:
        raise RuntimeError("Locked candidate fingerprint changed after historical confirmation.")
    if not SILVER_RAW_PATH.exists():
        raise FileNotFoundError(SILVER_RAW_PATH)

    training, feature_names = _load_training()
    spec = _ridge_alpha_100()
    exporter = FrozenRidgeExporter()
    cutoff = os.getenv("GITHUB_SHA", "local-shadow62-freeze")
    payload = exporter.export(
        spec,
        training,
        feature_names,
        SILVER_RAW_GIT_BLOB_SHA,
        cutoff,
    )
    payload["training_scope"] = (
        "historical Train+Validation+the already-confirmed one-shot historical Test; "
        "formal future holdout and shadow holdout excluded"
    )
    payload["candidate_id"] = CANDIDATE_ID
    payload["candidate_families"] = list(CANDIDATE_FAMILIES)
    payload["candidate_feature_fingerprint_sha256"] = candidate_fingerprint()
    payload["shadow_protocol_version"] = SHADOW_PROTOCOL_VERSION
    payload["model_payload_sha256"] = exporter.compute_hash(payload)

    model_path = OUTPUT_DIR / "models/xpt_xpd_62_ridge_alpha_100.json"
    exporter.write(payload, model_path)
    equivalence_error = _equivalence_error(spec, payload, training, feature_names)
    if equivalence_error > 1e-12:
        raise AssertionError(
            f"Shadow62 frozen payload is not numerically equivalent; max error={equivalence_error}"
        )

    baseline_path = Path("forward_holdout/models/ridge_alpha_100.json")
    baseline = FrozenRidgeRegressor.from_path(baseline_path)
    if tuple(feature_names[: len(baseline.feature_names)]) != baseline.feature_names:
        raise ValueError("Shadow62 first 52 features do not match the frozen live baseline order.")
    if tuple(feature_names[len(baseline.feature_names) :]) != CANDIDATE_FEATURES:
        raise ValueError("Shadow62 final 10 features do not match the locked candidate order.")

    source_hashes = {
        name: _sha256_file(ENHANCED_DIR / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }
    manifest = {
        "schema_version": 1,
        "freeze_id": SHADOW_FREEZE_ID,
        "protocol_version": SHADOW_PROTOCOL_VERSION,
        "protocol_predeclared_utc": "2026-08-13T06:34:00Z",
        "research_code_cutoff_commit": cutoff,
        "candidate_id": CANDIDATE_ID,
        "candidate_families": list(CANDIDATE_FAMILIES),
        "candidate_features": list(CANDIDATE_FEATURES),
        "candidate_feature_fingerprint_sha256": candidate_fingerprint(),
        "candidate_feature_count": len(CANDIDATE_FEATURES),
        "total_model_feature_count": len(feature_names),
        "historical_confirmation": {
            "status": "CONFIRMED",
            "run_id": HISTORICAL_CONFIRMATION_RUN_ID,
            "candidate_locked_before_old_test_read": True,
            "no_post_test_feature_tuning_allowed": True,
            "historical_test_reused_for_shadow_fit_only_after_confirmation": True,
            "historical_test_rescored_during_shadow_freeze": False,
        },
        "historical_source_dataset_path": str(SILVER_RAW_PATH),
        "historical_source_dataset_git_blob_sha": SILVER_RAW_GIT_BLOB_SHA,
        "enhanced_training_parquet_sha256": source_hashes,
        "training_scope": (
            "all labeled historical Train+Validation+confirmed historical Test rows available "
            "before shadow protocol; no future-holdout rows"
        ),
        "holdout_first_feature_bar_start_utc": SHADOW_FIRST_FEATURE_BAR_START_UTC.isoformat().replace("+00:00", "Z"),
        "holdout_last_feature_bar_start_exclusive_utc": SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC.isoformat().replace("+00:00", "Z"),
        "earliest_final_score_utc": SHADOW_EARLIEST_FINAL_SCORE_UTC.isoformat().replace("+00:00", "Z"),
        "minimum_exact_hour_outcomes": SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES,
        "fixed_window_days": SHADOW_FIXED_WINDOW_DAYS,
        "auxiliary_live_source": {
            "provider": "Dukascopy Public Historical Feed",
            "symbols": ["XPT.CMD/USD", "XPD.CMD/USD"],
            "offer_side": "BID",
            "timeframe": "1h",
            "timestamp_semantics": "BAR_START_UTC",
            "bar_value_availability": "KNOWN_AFTER_BAR_CLOSE",
            "exact_timestamp_alignment": True,
            "forward_fill": False,
            "interpolation": False,
        },
        "models": {
            "baseline": {
                "name": baseline.model_name,
                "path": str(baseline_path),
                "payload_sha256": baseline.model_payload_sha256,
                "feature_count": len(baseline.feature_names),
                "role": "unchanged frozen 52-feature live benchmark",
            },
            "candidate": {
                "name": spec.name,
                "candidate_id": CANDIDATE_ID,
                "path": str(model_path),
                "payload_sha256": payload["model_payload_sha256"],
                "feature_count": len(feature_names),
                "role": "research-only shadow candidate",
            },
        },
        "final_score_rules": {
            "one_shot_fixed_window": True,
            "no_interim_performance_metrics": True,
            "minimum_exact_hour_outcomes": SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES,
            "final_scorer_not_implemented_before_holdout_end": True,
        },
        "guardrails": {
            "research_only": True,
            "edge_status": "NOT_PROVEN",
            "live_model_mutated": False,
            "frozen_52_feature_graph_mutated": False,
            "automatic_live_promotion": False,
            "future_holdout_read": False,
            "interim_scoring_enabled": False,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = {
        "status": "PASS",
        "freeze_id": SHADOW_FREEZE_ID,
        "candidate_id": CANDIDATE_ID,
        "training_rows": int(len(training)),
        "training_first_timestamp_utc": training["timestamp_utc"].iloc[0].isoformat(),
        "training_last_timestamp_utc": training["timestamp_utc"].iloc[-1].isoformat(),
        "training_last_target_timestamp_utc": training["target_timestamp_utc"].iloc[-1].isoformat(),
        "feature_count": len(feature_names),
        "candidate_feature_count": len(CANDIDATE_FEATURES),
        "candidate_payload_sha256": payload["model_payload_sha256"],
        "max_prediction_equivalence_error": equivalence_error,
        "historical_test_rescored": False,
        "future_holdout_read": False,
        "performance_metrics_computed": False,
        "live_model_mutated": False,
    }
    (OUTPUT_DIR / "model_freeze_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
