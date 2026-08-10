"""Operational live-prediction platform for MetalPredictor.

The live package is intentionally separated from research/backtest modules. It reuses
only frozen model payloads and the frozen causal feature graph for inference.
"""

from metal_predictor.live.app import create_app

__all__ = ["create_app"]
