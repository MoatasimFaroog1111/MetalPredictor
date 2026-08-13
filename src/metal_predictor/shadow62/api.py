from __future__ import annotations

from fastapi import APIRouter, HTTPException

from metal_predictor.shadow62.contracts import ShadowRepository
from metal_predictor.shadow62.service import Shadow62Service


def create_shadow62_research_router(
    repository: ShadowRepository,
    service: Shadow62Service | None,
    *,
    enabled: bool,
    delay_minutes: int,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/research/shadow62", tags=["research-shadow62"])

    @router.get("/status")
    def status() -> dict[str, object]:
        if service is None:
            return {
                "component": "xpt-xpd-shadow-62-research",
                "collection_enabled": False,
                "delay_minutes": delay_minutes,
                "prediction_count": repository.prediction_count(),
                "outcome_count": repository.outcome_count(),
                "performance_metrics_available": False,
                "interim_scoring_enabled": False,
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
            }
        return {
            **service.status(),
            "collection_enabled": enabled,
            "delay_minutes": delay_minutes,
        }

    @router.get("/latest")
    def latest() -> dict[str, object]:
        snapshot = repository.latest_prediction()
        if snapshot is None:
            raise HTTPException(status_code=404, detail="No shadow62 observation has been recorded yet.")
        return snapshot.as_dict()

    return router
