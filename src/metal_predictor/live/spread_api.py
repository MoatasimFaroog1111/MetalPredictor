from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from metal_predictor.live.contracts import ForecastSnapshot
from metal_predictor.live.spread_profit import SpreadProfitAssumptions, SpreadProfitCalculator


class SpreadProfitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_spread_usd_per_kg: float = Field(ge=0)
    forecast_spread_usd_per_kg: float | None = Field(default=None, ge=0)
    target_profit_usd: float = Field(default=1.0, ge=0)
    fixed_round_trip_cost_usd: float = Field(default=0.0, ge=0)
    quantity_step_kg: float = Field(default=0.001, gt=0)
    minimum_trade_quantity_kg: float = Field(default=0.001, gt=0)

    def to_assumptions(self) -> SpreadProfitAssumptions:
        forecast_spread = (
            self.current_spread_usd_per_kg
            if self.forecast_spread_usd_per_kg is None
            else self.forecast_spread_usd_per_kg
        )
        return SpreadProfitAssumptions(
            current_spread_usd_per_kg=self.current_spread_usd_per_kg,
            forecast_spread_usd_per_kg=forecast_spread,
            target_profit_usd=self.target_profit_usd,
            fixed_round_trip_cost_usd=self.fixed_round_trip_cost_usd,
            quantity_step_kg=self.quantity_step_kg,
            minimum_trade_quantity_kg=self.minimum_trade_quantity_kg,
        )


def create_spread_research_router(
    latest_forecast: Callable[[], ForecastSnapshot | None],
    calculator: SpreadProfitCalculator | None = None,
) -> APIRouter:
    """Build an isolated research router around the pure spread calculator.

    The router depends only on a callable that supplies the latest frozen forecast,
    preserving dependency inversion and keeping market-data, inference, API transport,
    and spread arithmetic separate.
    """

    service = calculator or SpreadProfitCalculator()
    router = APIRouter(prefix="/api/v1/research", tags=["research"])

    @router.post("/spread-profit/latest")
    def latest_spread_profit(payload: SpreadProfitRequest) -> dict[str, object]:
        forecast = latest_forecast()
        if forecast is None:
            raise HTTPException(status_code=404, detail="No live forecast has been materialized yet.")
        try:
            return service.evaluate(forecast, payload.to_assumptions())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
