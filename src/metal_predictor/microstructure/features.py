from __future__ import annotations

import math

from metal_predictor.microstructure.contracts import (
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureFeatureVector,
    MicrostructureSnapshot,
)


class MicrostructureFeatureBuilder:
    """Build causal research features from one observed order-book snapshot.

    Only information available at ``snapshot.captured_at_utc`` or an earlier snapshot is
    used. The builder does not inspect forecasts, realized future returns, model outputs,
    or holdout data.
    """

    def build(
        self,
        snapshot: MicrostructureSnapshot,
        previous: MicrostructureSnapshot | None = None,
    ) -> MicrostructureFeatureVector:
        mid = snapshot.mid_usd_per_kg
        spread = snapshot.spread_usd_per_kg
        spread_bps = self._bps(spread, mid)
        top_bid_qty = snapshot.best_bid_quantity_kg
        top_ask_qty = snapshot.best_ask_quantity_kg
        top_total = top_bid_qty + top_ask_qty
        top_imbalance = (top_bid_qty - top_ask_qty) / top_total

        bid3 = self._depth_quantity(snapshot.bid_depth, 3)
        ask3 = self._depth_quantity(snapshot.ask_depth, 3)
        bid5 = self._depth_quantity(snapshot.bid_depth, 5)
        ask5 = self._depth_quantity(snapshot.ask_depth, 5)
        imbalance3 = self._imbalance(bid3, ask3)
        imbalance5 = self._imbalance(bid5, ask5)

        microprice = (
            snapshot.best_ask_usd_per_kg * top_bid_qty
            + snapshot.best_bid_usd_per_kg * top_ask_qty
        ) / top_total
        microprice_bias_bps = self._bps(microprice - mid, mid)

        bid_vwap5 = self._vwap(snapshot.bid_depth, 5)
        ask_vwap5 = self._vwap(snapshot.ask_depth, 5)
        bid_distance_bps = self._weighted_distance_bps(snapshot.bid_depth, mid, 5)
        ask_distance_bps = self._weighted_distance_bps(snapshot.ask_depth, mid, 5)
        bid_slope = self._book_slope_bps_per_kg(snapshot, side="bid", depth=5)
        ask_slope = self._book_slope_bps_per_kg(snapshot, side="ask", depth=5)

        values = {
            "spread_usd_per_kg": spread,
            "spread_bps": spread_bps,
            "top_bid_quantity_kg": top_bid_qty,
            "top_ask_quantity_kg": top_ask_qty,
            "top_quantity_imbalance": top_imbalance,
            "log_top_quantity_ratio": math.log(top_bid_qty / top_ask_qty),
            "bid_depth_3_kg": bid3,
            "ask_depth_3_kg": ask3,
            "depth_imbalance_3": imbalance3,
            "bid_depth_5_kg": bid5,
            "ask_depth_5_kg": ask5,
            "depth_imbalance_5": imbalance5,
            "microprice_usd_per_kg": microprice,
            "microprice_minus_mid_bps": microprice_bias_bps,
            "bid_vwap_5_usd_per_kg": bid_vwap5,
            "ask_vwap_5_usd_per_kg": ask_vwap5,
            "bid_depth_distance_bps": bid_distance_bps,
            "ask_depth_distance_bps": ask_distance_bps,
            "bid_slope_bps_per_kg": bid_slope,
            "ask_slope_bps_per_kg": ask_slope,
            "visible_bid_levels": float(len(snapshot.bid_depth)),
            "visible_ask_levels": float(len(snapshot.ask_depth)),
            "authenticated_quote_flag": float(
                snapshot.access_mode == "AUTHENTICATED_READ_ONLY"
            ),
            "public_cached_quote_flag": float(
                snapshot.freshness_status == "SERVER_CACHED_LESS_CURRENT"
            ),
        }
        values.update(
            self._temporal_features(
                snapshot=snapshot,
                previous=previous,
                spread_bps=spread_bps,
                imbalance5=imbalance5,
                microprice_bias_bps=microprice_bias_bps,
            )
        )
        return MicrostructureFeatureVector(
            captured_at_utc=snapshot.captured_at_utc,
            feature_version=MICROSTRUCTURE_FEATURE_VERSION,
            values=values,
        )

    def _temporal_features(
        self,
        *,
        snapshot: MicrostructureSnapshot,
        previous: MicrostructureSnapshot | None,
        spread_bps: float,
        imbalance5: float,
        microprice_bias_bps: float,
    ) -> dict[str, float]:
        defaults = {
            "has_previous_snapshot": 0.0,
            "seconds_since_previous": 0.0,
            "mid_log_return_since_previous": 0.0,
            "spread_bps_change": 0.0,
            "depth_imbalance_5_change": 0.0,
            "microprice_bias_bps_change": 0.0,
        }
        if previous is None:
            return defaults
        if (
            previous.security_id != snapshot.security_id
            or previous.currency != snapshot.currency
            or previous.captured_at_utc >= snapshot.captured_at_utc
        ):
            return defaults

        previous_spread_bps = self._bps(
            previous.spread_usd_per_kg,
            previous.mid_usd_per_kg,
        )
        previous_bid5 = self._depth_quantity(previous.bid_depth, 5)
        previous_ask5 = self._depth_quantity(previous.ask_depth, 5)
        previous_imbalance5 = self._imbalance(previous_bid5, previous_ask5)
        previous_top_total = (
            previous.best_bid_quantity_kg + previous.best_ask_quantity_kg
        )
        previous_microprice = (
            previous.best_ask_usd_per_kg * previous.best_bid_quantity_kg
            + previous.best_bid_usd_per_kg * previous.best_ask_quantity_kg
        ) / previous_top_total
        previous_microprice_bias_bps = self._bps(
            previous_microprice - previous.mid_usd_per_kg,
            previous.mid_usd_per_kg,
        )
        return {
            "has_previous_snapshot": 1.0,
            "seconds_since_previous": (
                snapshot.captured_at_utc - previous.captured_at_utc
            ).total_seconds(),
            "mid_log_return_since_previous": math.log(
                snapshot.mid_usd_per_kg / previous.mid_usd_per_kg
            ),
            "spread_bps_change": spread_bps - previous_spread_bps,
            "depth_imbalance_5_change": imbalance5 - previous_imbalance5,
            "microprice_bias_bps_change": (
                microprice_bias_bps - previous_microprice_bias_bps
            ),
        }

    @staticmethod
    def _bps(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) * 10_000.0

    @staticmethod
    def _depth_quantity(levels, depth: int) -> float:
        return float(sum(level.quantity_kg for level in levels[:depth]))

    @staticmethod
    def _imbalance(bid_quantity: float, ask_quantity: float) -> float:
        total = bid_quantity + ask_quantity
        return (bid_quantity - ask_quantity) / total if total > 0 else 0.0

    @staticmethod
    def _vwap(levels, depth: int) -> float:
        chosen = levels[:depth]
        quantity = sum(level.quantity_kg for level in chosen)
        if quantity <= 0:
            raise ValueError("Visible market depth quantity must be positive.")
        return float(
            sum(level.price_usd_per_kg * level.quantity_kg for level in chosen)
            / quantity
        )

    @staticmethod
    def _weighted_distance_bps(levels, mid: float, depth: int) -> float:
        chosen = levels[:depth]
        quantity = sum(level.quantity_kg for level in chosen)
        if quantity <= 0:
            raise ValueError("Visible market depth quantity must be positive.")
        weighted_distance = sum(
            abs(level.price_usd_per_kg - mid) * level.quantity_kg
            for level in chosen
        ) / quantity
        return weighted_distance / mid * 10_000.0

    @staticmethod
    def _book_slope_bps_per_kg(
        snapshot: MicrostructureSnapshot,
        *,
        side: str,
        depth: int,
    ) -> float:
        levels = snapshot.bid_depth if side == "bid" else snapshot.ask_depth
        chosen = levels[:depth]
        if len(chosen) < 2:
            return 0.0
        cumulative_quantity = sum(level.quantity_kg for level in chosen)
        if cumulative_quantity <= 0:
            return 0.0
        if side == "bid":
            price_distance = chosen[0].price_usd_per_kg - chosen[-1].price_usd_per_kg
        else:
            price_distance = chosen[-1].price_usd_per_kg - chosen[0].price_usd_per_kg
        distance_bps = price_distance / snapshot.mid_usd_per_kg * 10_000.0
        return distance_bps / cumulative_quantity
