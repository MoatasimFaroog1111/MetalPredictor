from __future__ import annotations

import hashlib
import json
from pathlib import Path

from metal_predictor.multi_horizon import (
    BullionVaultChartCsvAuditor,
    BullionVaultChartCsvLoader,
    DatasetState,
    HORIZONS,
    ResearchGuardrails,
    get_horizon,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "research_data" / "bullionvault_horizons" / "manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_horizon_registry_is_explicit_and_daily_remains_pending() -> None:
    assert [spec.key for spec in HORIZONS] == ["4h", "12h", "1d", "2d", "30d"]
    assert [spec.interval_seconds for spec in HORIZONS] == [
        14_400,
        43_200,
        86_400,
        172_800,
        2_592_000,
    ]
    assert get_horizon("1d").dataset_state is DatasetState.DATA_PENDING
    assert get_horizon("4H").route == "/forecast/4h"


def test_guardrails_keep_multi_horizon_research_out_of_live_execution() -> None:
    guards = ResearchGuardrails()
    assert guards.edge_status == "NOT_PROVEN"
    assert guards.research_only is True
    assert guards.buy_sell_enabled is False
    assert guards.execution_enabled is False
    assert guards.live_model_mutated is False
    assert guards.frozen_52_feature_graph_mutated is False


def test_raw_exports_match_locked_manifest_without_resampling_or_fill() -> None:
    manifest = _manifest()
    auditor = BullionVaultChartCsvAuditor()
    datasets = manifest["datasets"]

    for key in ("4h", "12h", "2d", "30d"):
        entry = datasets[key]
        path = ROOT / entry["raw_path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]

        audit = auditor.audit(
            path,
            expected_interval_seconds=int(entry["interval_seconds"]),
        )
        assert audit.sha256 == entry["sha256"]
        assert audit.row_count == entry["row_count"]
        assert audit.exact_interval_pass is True
        assert audit.observed_interval_seconds == (entry["interval_seconds"],)
        assert audit.timestamp_timezone_semantics == "UNVERIFIED_EXPORT_CLIENT_TIMEZONE"
        assert audit.newest_bucket_status == "POTENTIALLY_INCOMPLETE"


def test_loader_excludes_potentially_incomplete_newest_bar_by_default() -> None:
    manifest = _manifest()
    loader = BullionVaultChartCsvLoader()

    for key in ("4h", "12h", "2d", "30d"):
        entry = manifest["datasets"][key]
        loaded = loader.load(
            ROOT / entry["raw_path"],
            expected_interval_seconds=int(entry["interval_seconds"]),
        )
        assert loaded.potentially_incomplete_newest_excluded is True
        assert len(loaded.frame) == int(entry["row_count"]) - 1
        assert loaded.frame["timestamp_source"].is_monotonic_increasing
        assert set(loaded.frame.columns) == {
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
            "timestamp_source",
        }


def test_daily_manifest_has_no_synthetic_raw_file() -> None:
    daily = _manifest()["datasets"]["1d"]
    assert daily["state"] == "DATA_PENDING"
    assert daily["interval_seconds"] == 86_400
    assert daily["raw_path"] is None
    assert daily["sha256"] is None
    assert daily["row_count"] == 0
