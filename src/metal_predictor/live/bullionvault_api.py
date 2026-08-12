from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from metal_predictor.live.contracts import ForecastSnapshot
from metal_predictor.live.quote_contracts import MarketQuoteProvider
from metal_predictor.live.spread_profit import (
    MarketQuoteProfitAssumptions,
    SpreadProfitCalculator,
)


class BullionVaultProfitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_spread_usd_per_kg: float | None = Field(default=None, ge=0)
    target_profit_usd: float = Field(default=1.0, ge=0)
    fixed_round_trip_cost_usd: float = Field(default=0.0, ge=0)
    quantity_step_kg: float = Field(default=0.001, gt=0)
    minimum_trade_quantity_kg: float = Field(default=0.001, gt=0)

    def to_assumptions(self) -> MarketQuoteProfitAssumptions:
        return MarketQuoteProfitAssumptions(
            forecast_spread_usd_per_kg=self.forecast_spread_usd_per_kg,
            target_profit_usd=self.target_profit_usd,
            fixed_round_trip_cost_usd=self.fixed_round_trip_cost_usd,
            quantity_step_kg=self.quantity_step_kg,
            minimum_trade_quantity_kg=self.minimum_trade_quantity_kg,
        )


def create_bullionvault_research_router(
    latest_forecast: Callable[[], ForecastSnapshot | None],
    quote_provider: MarketQuoteProvider,
    calculator: SpreadProfitCalculator | None = None,
) -> APIRouter:
    """Expose BullionVault quotes without coupling API transport to provider internals."""

    service = calculator or SpreadProfitCalculator()
    router = APIRouter(prefix="/api/v1/research/bullionvault", tags=["research"])

    @router.get("/quote")
    def latest_quote() -> dict[str, object]:
        try:
            quote = quote_provider.fetch_quote()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "quote": quote.as_dict(),
            "safety": {
                "read_only": True,
                "execution_enabled": False,
                "order_submission_available": False,
            },
        }

    @router.post("/spread-profit/latest")
    def latest_spread_profit(payload: BullionVaultProfitRequest) -> dict[str, object]:
        forecast = latest_forecast()
        if forecast is None:
            raise HTTPException(status_code=404, detail="No live forecast has been materialized yet.")
        try:
            quote = quote_provider.fetch_quote()
            return service.evaluate_market_quote(
                forecast,
                quote,
                payload.to_assumptions(),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
