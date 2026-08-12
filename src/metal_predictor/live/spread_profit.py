from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math

from metal_predictor.live.contracts import ForecastSnapshot


@dataclass(frozen=True)
class SpreadProfitAssumptions:
    """User-supplied execution-cost assumptions for research simulation only.

    The live Gold API feed currently supplies a reference price, not executable bid/ask
    quotes. Spreads are therefore explicit assumptions and must never be presented as
    broker quotes. The calculator is intentionally independent of any market provider so
    a real quote adapter can be introduced later without changing the calculation rules.
    """

    current_spread_usd_per_kg: float
    forecast_spread_usd_per_kg: float
    target_profit_usd: float = 1.0
    fixed_round_trip_cost_usd: float = 0.0
    quantity_step_kg: float = 0.001
    minimum_trade_quantity_kg: float = 0.001

    def __post_init__(self) -> None:
        finite_non_negative = {
            "current_spread_usd_per_kg": self.current_spread_usd_per_kg,
            "forecast_spread_usd_per_kg": self.forecast_spread_usd_per_kg,
            "target_profit_usd": self.target_profit_usd,
            "fixed_round_trip_cost_usd": self.fixed_round_trip_cost_usd,
        }
        for name, value in finite_non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name, value in {
            "quantity_step_kg": self.quantity_step_kg,
            "minimum_trade_quantity_kg": self.minimum_trade_quantity_kg,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and greater than zero.")


@dataclass(frozen=True)
class SideResearchScenario:
    side: str
    expected_margin_usd_per_kg: float
    positive_margin_after_assumed_costs: bool
    minimum_quantity_for_target_profit_kg: float | None
    expected_profit_at_minimum_quantity_usd: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "expected_margin_usd_per_kg": self.expected_margin_usd_per_kg,
            "positive_margin_after_assumed_costs": self.positive_margin_after_assumed_costs,
            "minimum_quantity_for_target_profit_kg": self.minimum_quantity_for_target_profit_kg,
            "expected_profit_at_minimum_quantity_usd": self.expected_profit_at_minimum_quantity_usd,
        }


@dataclass(frozen=True)
class ModelSpreadScenario:
    model: str
    direction: str
    predicted_mid_usd_per_kg: float
    predicted_bid_usd_per_kg: float
    predicted_ask_usd_per_kg: float
    long: SideResearchScenario
    short: SideResearchScenario

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "direction": self.direction,
            "predicted_mid_usd_per_kg": self.predicted_mid_usd_per_kg,
            "predicted_bid_usd_per_kg": self.predicted_bid_usd_per_kg,
            "predicted_ask_usd_per_kg": self.predicted_ask_usd_per_kg,
            "long": self.long.as_dict(),
            "short": self.short.as_dict(),
        }


class SpreadProfitCalculator:
    """Pure calculator for spread-aware next-hour research scenarios.

    No order creation, execution, position sizing recommendation, model mutation, or
    performance scoring occurs here. The service only translates a frozen forecast plus
    explicit cost assumptions into transparent arithmetic for both long and short paths.
    """

    def evaluate(
        self,
        forecast: ForecastSnapshot,
        assumptions: SpreadProfitAssumptions,
    ) -> dict[str, object]:
        current_mid = self._positive_price(
            forecast.current_price_usd_per_kg,
            "current_price_usd_per_kg",
        )
        current_half_spread = assumptions.current_spread_usd_per_kg / 2.0
        forecast_half_spread = assumptions.forecast_spread_usd_per_kg / 2.0
        current_bid = current_mid - current_half_spread
        current_ask = current_mid + current_half_spread
        if current_bid <= 0:
            raise ValueError("Current spread assumption makes the bid non-positive.")

        baseline = self._model_scenario(
            model=forecast.baseline_model,
            direction=forecast.baseline_direction,
            predicted_mid=forecast.baseline_predicted_price_usd_per_kg,
            current_bid=current_bid,
            current_ask=current_ask,
            forecast_half_spread=forecast_half_spread,
            assumptions=assumptions,
        )
        challenger = self._model_scenario(
            model=forecast.challenger_model,
            direction=forecast.challenger_direction,
            predicted_mid=forecast.challenger_predicted_price_usd_per_kg,
            current_bid=current_bid,
            current_ask=current_ask,
            forecast_half_spread=forecast_half_spread,
            assumptions=assumptions,
        )

        serialized = forecast.as_dict()
        return {
            "reference": {
                "current_mid_usd_per_kg": current_mid,
                "current_bid_usd_per_kg": current_bid,
                "current_ask_usd_per_kg": current_ask,
                "current_price_time_utc": serialized.get("current_price_time_utc"),
                "current_price_time_saudi": serialized.get("current_price_time_saudi"),
                "forecast_target_time_utc": serialized.get("forecast_target_time_utc"),
                "forecast_target_time_saudi": serialized.get("forecast_target_time_saudi"),
            },
            "assumptions": {
                "spread_source": "USER_ASSUMPTION",
                "spread_quote_status": "NOT_EXECUTABLE_MARKET_QUOTE",
                "current_spread_usd_per_kg": assumptions.current_spread_usd_per_kg,
                "forecast_spread_usd_per_kg": assumptions.forecast_spread_usd_per_kg,
                "target_profit_usd": assumptions.target_profit_usd,
                "fixed_round_trip_cost_usd": assumptions.fixed_round_trip_cost_usd,
                "quantity_step_kg": assumptions.quantity_step_kg,
                "minimum_trade_quantity_kg": assumptions.minimum_trade_quantity_kg,
            },
            "baseline": baseline.as_dict(),
            "challenger": challenger.as_dict(),
            "safety": {
                "edge_status": forecast.edge_status,
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
            },
        }

    def _model_scenario(
        self,
        *,
        model: str,
        direction: str,
        predicted_mid: float,
        current_bid: float,
        current_ask: float,
        forecast_half_spread: float,
        assumptions: SpreadProfitAssumptions,
    ) -> ModelSpreadScenario:
        predicted_mid = self._positive_price(predicted_mid, "predicted_mid_usd_per_kg")
        predicted_bid = predicted_mid - forecast_half_spread
        predicted_ask = predicted_mid + forecast_half_spread
        if predicted_bid <= 0:
            raise ValueError("Forecast spread assumption makes the predicted bid non-positive.")

        # Long path: buy at the current ask, then sell at the forecast bid.
        long_margin = predicted_bid - current_ask
        # Short path: sell at the current bid, then buy back at the forecast ask.
        short_margin = current_bid - predicted_ask

        return ModelSpreadScenario(
            model=model,
            direction=direction,
            predicted_mid_usd_per_kg=predicted_mid,
            predicted_bid_usd_per_kg=predicted_bid,
            predicted_ask_usd_per_kg=predicted_ask,
            long=self._side_scenario("LONG", long_margin, assumptions),
            short=self._side_scenario("SHORT", short_margin, assumptions),
        )

    def _side_scenario(
        self,
        side: str,
        margin_per_kg: float,
        assumptions: SpreadProfitAssumptions,
    ) -> SideResearchScenario:
        if margin_per_kg <= 0:
            return SideResearchScenario(
                side=side,
                expected_margin_usd_per_kg=margin_per_kg,
                positive_margin_after_assumed_costs=False,
                minimum_quantity_for_target_profit_kg=None,
                expected_profit_at_minimum_quantity_usd=None,
            )

        required_before_rounding = (
            assumptions.target_profit_usd + assumptions.fixed_round_trip_cost_usd
        ) / margin_per_kg
        quantity = max(
            assumptions.minimum_trade_quantity_kg,
            self._ceil_to_step(required_before_rounding, assumptions.quantity_step_kg),
        )
        profit = quantity * margin_per_kg - assumptions.fixed_round_trip_cost_usd
        return SideResearchScenario(
            side=side,
            expected_margin_usd_per_kg=margin_per_kg,
            positive_margin_after_assumed_costs=(profit > 0),
            minimum_quantity_for_target_profit_kg=quantity,
            expected_profit_at_minimum_quantity_usd=profit,
        )

    @staticmethod
    def _ceil_to_step(value: float, step: float) -> float:
        value_decimal = Decimal(str(max(0.0, value)))
        step_decimal = Decimal(str(step))
        units = (value_decimal / step_decimal).to_integral_value(rounding=ROUND_CEILING)
        return float(units * step_decimal)

    @staticmethod
    def _positive_price(value: float, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        return numeric
