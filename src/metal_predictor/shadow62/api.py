from __future__ import annotations

from fastapi import APIRouter

from metal_predictor.shadow62.contracts import ShadowRepository
from metal_predictor.shadow62.service import Shadow62Service


def create_shadow62_research_router(
    repository: ShadowRepository,
    service: Shadow62Service | None,
    *,
    enabled: bool,
    delay_minutes: int,
) -> APIRouter:
    """Expose operational status only; prediction values stay sealed during holdout."""

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
                "prediction_values_exposed": False,
                "outcome_values_exposed": False,
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
            "prediction_values_exposed": False,
            "outcome_values_exposed": False,
        }

    return router
