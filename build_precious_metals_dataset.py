from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.precious_metals.contracts import PALLADIUM, PLATINUM, PreciousMetalInstrument
from metal_predictor.precious_metals.dukascopy_public_source import (
    DukascopyPublicH1UrlPlanner,
    DukascopyPublicHistoricalMetalSource,
)
from metal_predictor.precious_metals.provenance import (
    HistoricalBootstrapManifest,
    HistoricalBootstrapProvenanceGate,
)


SILVER_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
OUTPUT_DIR = Path("data/market")


def silver_window(path: Path = SILVER_PATH) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.read_parquet(path, columns=["timestamp_utc"])["timestamp_utc"]
    timestamps = pd.to_datetime(timestamps, utc=True, errors="raise")
    return timestamps.min(), timestamps.max()


def _provenance_manifest(
    instrument: PreciousMetalInstrument,
    requested_history_days: int,
) -> HistoricalBootstrapManifest:
    return HistoricalBootstrapManifest(
        asset=instrument.asset,
        provider="Dukascopy Public Historical Feed",
        source_symbol=instrument.dukascopy_name,
        instrument_semantics="COMMODITY_CFD_CROSS_FEED",
        interval="1h",
        requested_history_days=requested_history_days,
        has_open=True,
        has_high=True,
        has_low=True,
        has_close=True,
        exact_utc_hour_guaranteed=True,
        forward_fill_used=False,
        interpolation_used=False,
        provenance_manifest_embedded=True,
        source_repository="Dukascopy public historical feed",
        source_path="/v1/candles/hour/{instrument}/BID/{year}/{month}",
    )


def _write_instrument(
    frame: pd.DataFrame,
    instrument: PreciousMetalInstrument,
    manifest: HistoricalBootstrapManifest,
    provenance_assessment: dict[str, object],
) -> dict[str, object]:
    if frame.empty:
        raise RuntimeError(f"Dukascopy returned no H1 rows for {instrument.dukascopy_name}.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / f"{instrument.output_stem}.parquet"
    csv_path = OUTPUT_DIR / f"{instrument.output_stem}.csv"
    report_path = OUTPUT_DIR / f"{instrument.output_stem}_quality_report.json"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)

    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    spec = DukascopyPublicH1UrlPlanner.feed_spec(instrument)
    report = {
        "status": "SOURCE_PARSE_COMPLETE_UNVALIDATED_FOR_MODEL",
        "model_readiness": "PENDING_DEVELOPMENT_COVERAGE_GATE",
        "asset": instrument.asset,
        "source_provider": "Dukascopy Public Historical Feed",
        "source_symbol": instrument.dukascopy_name,
        "source_feed_code": spec.feed_code,
        "source_contract": "commodity CFD, 1 CFD = 1 troy ounce",
        "source_quote_unit": "USD/troy_oz",
        "normalized_unit": "USD/kg",
        "timeframe": "1hour",
        "offer_side": "Bid",
        "authentication_required": False,
        "provider_first_h1_utc": spec.earliest_h1_utc.isoformat(),
        "rows": int(len(frame)),
        "first_timestamp_utc": timestamps.iloc[0].isoformat(),
        "last_timestamp_utc": timestamps.iloc[-1].isoformat(),
        "exact_hour_rows": int(
            (
                timestamps.dt.minute.eq(0)
                & timestamps.dt.second.eq(0)
                & timestamps.dt.microsecond.eq(0)
            ).sum()
        ),
        "provenance_manifest": manifest.as_dict(),
        "provenance_gate": provenance_assessment,
        "source_policy": {
            "exact_timestamp_alignment": True,
            "source_gaps_preserved": True,
            "synthetic_flat_candles": False,
            "forward_fill": False,
            "backward_fill": False,
            "interpolation": False,
            "nearest_timestamp_match": False,
            "conflicting_duplicate_hours": "fail_closed",
            "sparse_provider_coverage": (
                "must_pass_pre_registered_fold_coverage_gate_before_model_fit"
            ),
        },
        "outputs": [str(parquet_path), str(csv_path)],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build() -> dict[str, object]:
    start, end = silver_window()
    requested_history_days = int((end.normalize() - start.normalize()).days + 1)
    instruments = (PLATINUM, PALLADIUM)
    manifests = tuple(
        _provenance_manifest(instrument, requested_history_days)
        for instrument in instruments
    )
    assessments = HistoricalBootstrapProvenanceGate().validate_many(manifests)
    assessments_by_asset = {item.asset: item.as_dict() for item in assessments}

    source = DukascopyPublicHistoricalMetalSource()
    reports = []
    for instrument, manifest in zip(instruments, manifests, strict=True):
        frame = source.fetch_hourly(
            instrument,
            start.to_pydatetime(),
            end.to_pydatetime(),
        )
        reports.append(
            _write_instrument(
                frame,
                instrument,
                manifest,
                assessments_by_asset[instrument.asset],
            )
        )

    combined = {
        "status": "SOURCE_ACQUIRED_UNVALIDATED_FOR_MODEL",
        "model_readiness": "PENDING_DEVELOPMENT_COVERAGE_GATE",
        "source_access": "KEYLESS_READ_ONLY_PUBLIC_HISTORY",
        "provenance_gate": {
            "status": "PASS",
            "validated_before_source_fetch": True,
            "assessments": [item.as_dict() for item in assessments],
        },
        "research_only": True,
        "model_mutated": False,
        "frozen_feature_graph_mutated": False,
        "future_holdout_read": False,
        "window": {"start_utc": start.isoformat(), "end_utc": end.isoformat()},
        "requested_history_days": requested_history_days,
        "instruments": reports,
    }
    (OUTPUT_DIR / "precious_metals_source_report.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    return combined


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
