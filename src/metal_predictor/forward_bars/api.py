from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from metal_predictor.forward_bars.contracts import FORWARD_HORIZON_SECONDS
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository


def create_forward_bar_research_router(
    repository: SQLiteForwardBarRepository,
    *,
    enabled: bool,
    source_collection_enabled: bool,
    materialization_interval_seconds: int,
    close_delay_seconds: int,
    source_cadence_seconds: int,
    security_id: str,
    currency: str,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/research/forward-bars", tags=["research"])

    @router.get("/status")
    def status() -> dict[str, object]:
        stored = repository.status_snapshot()
        horizons: dict[str, object] = {}
        for key in FORWARD_HORIZON_SECONDS:
            horizons[key] = stored.get(
                key,
                {"bar_count": 0, "gap_count": 0, "latest_assessed_end_utc": None, "latest_bar": None},
            )
        return {
            "component": "bullionvault-forward-multi-horizon-data-factory",
            "bar_version": "bullionvault-forward-multi-horizon-v1",
            "source_stream": "BULLIONVAULT_READ_ONLY_MICROSTRUCTURE_SNAPSHOTS",
            "collection_enabled": bool(enabled),
            "source_snapshot_collection_enabled": bool(source_collection_enabled),
            "materialization_interval_seconds": int(materialization_interval_seconds),
            "close_delay_seconds": int(close_delay_seconds),
            "source_cadence_seconds": int(source_cadence_seconds),
            "security_id": security_id,
            "currency": currency,
            "horizon_intervals_seconds": dict(FORWARD_HORIZON_SECONDS),
            "horizons": horizons,
            "safety": {
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
                "shadow62_mutated": False,
                "historical_chart_data_merged": False,
                "fill_or_interpolation_used": False,
            },
        }

    @router.get("/latest")
    def latest(horizon: str = "4h") -> dict[str, object]:
        key = horizon.strip().lower()
        if key not in FORWARD_HORIZON_SECONDS:
            raise HTTPException(status_code=422, detail="Unsupported forward-bar horizon.")
        bar = repository.latest_bar(key)
        if bar is None:
            raise HTTPException(status_code=404, detail="No completed observed forward bar yet.")
        return bar.as_dict()

    @router.get("/history")
    def history(horizon: str = "4h", limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, object]]:
        key = horizon.strip().lower()
        if key not in FORWARD_HORIZON_SECONDS:
            raise HTTPException(status_code=422, detail="Unsupported forward-bar horizon.")
        return [bar.as_dict() for bar in repository.history(key, limit=limit)]

    return router
