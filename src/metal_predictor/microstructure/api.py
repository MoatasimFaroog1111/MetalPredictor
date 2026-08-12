from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from metal_predictor.microstructure.contracts import MicrostructureRepository


SAFETY = {
    "research_only": True,
    "model_mutated": False,
    "frozen_feature_graph_mutated": False,
    "buy_sell_enabled": False,
    "execution_enabled": False,
    "order_submission_available": False,
}


def create_microstructure_research_router(
    repository: MicrostructureRepository,
    *,
    collection_enabled: bool,
    interval_seconds: int,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/research/microstructure", tags=["research"])

    @router.get("/status")
    def status() -> dict[str, object]:
        latest = repository.latest_record()
        return {
            "component": "bullionvault-microstructure-research",
            "collection_enabled": bool(collection_enabled),
            "interval_seconds": int(interval_seconds),
            "snapshot_count": repository.count(),
            "latest_captured_at_utc": (
                latest.snapshot.captured_at_utc.isoformat() if latest else None
            ),
            "feature_version": (
                latest.features.feature_version if latest else "bullionvault-microstructure-v1"
            ),
            "safety": dict(SAFETY),
        }

    @router.get("/latest")
    def latest() -> dict[str, object]:
        record = repository.latest_record()
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="No BullionVault microstructure research snapshot has been collected yet.",
            )
        return record.as_dict()

    @router.get("/history")
    def history(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> list[dict[str, object]]:
        return [record.as_dict() for record in repository.recent_records(limit=limit)]

    return router
