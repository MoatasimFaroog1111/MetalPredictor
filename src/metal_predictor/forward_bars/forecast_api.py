from __future__ import annotations

from fastapi import APIRouter, HTTPException

from metal_predictor.forward_bars.forecasting import MultiHorizonBaselineForecastService


def create_multi_horizon_forecast_router(
    service: MultiHorizonBaselineForecastService,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/research/multi-horizon-forecast",
        tags=["research-multi-horizon-forecast"],
    )

    @router.get("/status")
    def status() -> dict[str, object]:
        return service.status()

    @router.get("/{horizon_key}")
    def forecast(horizon_key: str) -> dict[str, object]:
        try:
            return service.forecast(horizon_key)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
