from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math

from metal_predictor.live.contracts import ForecastSnapshot
from metal_predictor.live.quote_contracts import BidAskMarketQuote


@dataclass(frozen=True)
class SpreadProfitAssumptions:
    """User-supplied execution-cost assumptions for research simulation only.

    The calculator is intentionally independent of any market provider. Manual spread
    assumptions remain available as a fallback even when a live quote adapter is present.
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
class MarketQuoteProfitAssumptions:
    """Assumptions needed after the current bid/ask comes from a market quote provider."""

    forecast_spread_usd_per_kg: float | None = None
    target_profit_usd: float = 1.0
    fixed_round_trip_cost_usd: float = 0.0
    quantity_step_kg: float = 0.001
    minimum_trade_quantity_kg: float = 0.001

    def __post_init__(self) -> None:
        if self.forecast_spread_usd_per_kg is not None:
            value = float(self.forecast_spread_usd_per_kg)
            if not math.isfinite(value) or value < 0:
                raise ValueError("forecast_spread_usd_per_kg must be finite and non-negative.")
        for name, value in {
            "target_profit_usd": self.target_profit_usd,
            "fixed_round_trip_cost_usd": self.fixed_round_trip_cost_usd,
        }.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name, value in {
            "quantity_step_kg": self.quantity_step_kg,
            "minimum_trade_quantity_kg": self.minimum_trade_quantity_kg,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and greater than zero.")

    def with_current_spread(self, current_spread_usd_per_kg: float) -> SpreadProfitAssumptions:
        forecast_spread = (
            current_spread_usd_per_kg
            if self.forecast_spread_usd_per_kg is None
            else self.forecast_spread_usd_per_kg
        )
        return SpreadProfitAssumptions(
            current_spread_usd_per_kg=current_spread_usd_per_kg,
            forecast_spread_usd_per_kg=forecast_spread,
            target_profit_usd=self.target_profit_usd,
            fixed_round_trip_cost_usd=self.fixed_round_trip_cost_usd,
            quantity_step_kg=self.quantity_step_kg,
            minimum_trade_quantity_kg=self.minimum_trade_quantity_kg,
        )


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
    performance scoring occurs here. Live quotes may be injected through a provider-neutral
    contract; the calculator never performs network I/O itself.
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
            "safety": self._safety(forecast),
        }

    def evaluate_market_quote(
        self,
        forecast: ForecastSnapshot,
        quote: BidAskMarketQuote,
        assumptions: MarketQuoteProfitAssumptions,
    ) -> dict[str, object]:
        """Apply frozen model returns to a provider's current executable-market midpoint.

        The model was trained to predict next-hour log return, not a BullionVault absolute
        price level. Applying the frozen return to the BullionVault midpoint keeps the
        execution venue's current basis while preserving the model output unchanged.
        """

        current_bid = self._positive_price(quote.best_bid_usd_per_kg, "best_bid_usd_per_kg")
        current_ask = self._positive_price(quote.best_ask_usd_per_kg, "best_ask_usd_per_kg")
        if current_ask <= current_bid:
            raise ValueError("Market quote must have ask above bid.")
        current_mid = quote.mid_usd_per_kg
        spread_assumptions = assumptions.with_current_spread(quote.spread_usd_per_kg)
        forecast_half_spread = spread_assumptions.forecast_spread_usd_per_kg / 2.0

        baseline = self._model_scenario(
            model=forecast.baseline_model,
            direction=forecast.baseline_direction,
            predicted_mid=self._project_mid(current_mid, forecast.baseline_log_return_1h),
            current_bid=current_bid,
            current_ask=current_ask,
            forecast_half_spread=forecast_half_spread,
            assumptions=spread_assumptions,
        )
        challenger = self._model_scenario(
            model=forecast.challenger_model,
            direction=forecast.challenger_direction,
            predicted_mid=self._project_mid(current_mid, forecast.challenger_log_return_1h),
            current_bid=current_bid,
            current_ask=current_ask,
            forecast_half_spread=forecast_half_spread,
            assumptions=spread_assumptions,
        )

        baseline_data = self._with_entry_liquidity(baseline, quote)
        challenger_data = self._with_entry_liquidity(challenger, quote)
        serialized = forecast.as_dict()
        forecast_spread_source = (
            "CURRENT_BULLIONVAULT_SPREAD_CARRY_FORWARD"
            if assumptions.forecast_spread_usd_per_kg is None
            else "USER_ASSUMPTION"
        )
        model_reference = self._positive_price(
            forecast.current_price_usd_per_kg,
            "current_price_usd_per_kg",
        )

        return {
            "reference": {
                "current_mid_usd_per_kg": current_mid,
                "current_bid_usd_per_kg": current_bid,
                "current_ask_usd_per_kg": current_ask,
                "current_spread_usd_per_kg": quote.spread_usd_per_kg,
                "best_bid_quantity_kg": quote.best_bid_quantity_kg,
                "best_ask_quantity_kg": quote.best_ask_quantity_kg,
                "quote_fetched_at_utc": quote.fetched_at_utc.isoformat(),
                "current_price_time_utc": quote.fetched_at_utc.isoformat(),
                "forecast_target_time_utc": serialized.get("forecast_target_time_utc"),
                "forecast_target_time_saudi": serialized.get("forecast_target_time_saudi"),
                "model_reference_price_usd_per_kg": model_reference,
                "quote_vs_model_basis_usd_per_kg": current_mid - model_reference,
            },
            "market_quote": quote.as_dict(),
            "assumptions": {
                "spread_source": quote.source_provider,
                "spread_quote_status": quote.freshness_status,
                "current_spread_usd_per_kg": quote.spread_usd_per_kg,
                "forecast_spread_usd_per_kg": spread_assumptions.forecast_spread_usd_per_kg,
                "forecast_spread_source": forecast_spread_source,
                "target_profit_usd": spread_assumptions.target_profit_usd,
                "fixed_round_trip_cost_usd": spread_assumptions.fixed_round_trip_cost_usd,
                "quantity_step_kg": spread_assumptions.quantity_step_kg,
                "minimum_trade_quantity_kg": spread_assumptions.minimum_trade_quantity_kg,
                "provider_commission_included": False,
                "future_market_depth_known": False,
            },
            "projection": {
                "method": "APPLY_FROZEN_LOG_RETURN_TO_BULLIONVAULT_MID",
                "model_mutated": False,
                "forecast_horizon_hours": 1,
            },
            "baseline": baseline_data,
            "challenger": challenger_data,
            "safety": self._safety(forecast),
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

        long_margin = predicted_bid - current_ask
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
    def _with_entry_liquidity(
        scenario: ModelSpreadScenario,
        quote: BidAskMarketQuote,
    ) -> dict[str, object]:
        data = scenario.as_dict()
        long_min = scenario.long.minimum_quantity_for_target_profit_kg
        short_min = scenario.short.minimum_quantity_for_target_profit_kg
        data["long"]["current_entry_top_of_book_quantity_kg"] = quote.best_ask_quantity_kg
        data["long"]["current_entry_top_of_book_sufficient"] = (
            None if long_min is None else long_min <= quote.best_ask_quantity_kg
        )
        data["short"]["current_entry_top_of_book_quantity_kg"] = quote.best_bid_quantity_kg
        data["short"]["current_entry_top_of_book_sufficient"] = (
            None if short_min is None else short_min <= quote.best_bid_quantity_kg
        )
        data["long"]["future_exit_depth_known"] = False
        data["short"]["future_exit_depth_known"] = False
        return data

    @staticmethod
    def _project_mid(current_mid: float, predicted_log_return_1h: float) -> float:
        log_return = float(predicted_log_return_1h)
        if not math.isfinite(log_return):
            raise ValueError("Predicted log return must be finite.")
        try:
            projected = current_mid * math.exp(log_return)
        except OverflowError:
            raise ValueError("Predicted log return produces an invalid projected price.") from None
        return SpreadProfitCalculator._positive_price(projected, "projected_mid_usd_per_kg")

    @staticmethod
    def _safety(forecast: ForecastSnapshot) -> dict[str, object]:
        return {
            "edge_status": forecast.edge_status,
            "research_only": True,
            "buy_sell_enabled": False,
            "execution_enabled": False,
            "order_submission_available": False,
        }

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
