from __future__ import annotations

import asyncio
import logging

from metal_predictor.forward_bars.factory import ForwardBarFactory


logger = logging.getLogger(__name__)


class ForwardBarMaterializationScheduler:
    def __init__(self, factory: ForwardBarFactory, *, interval_seconds: int = 60) -> None:
        interval = int(interval_seconds)
        if not 30 <= interval <= 3600:
            raise ValueError("Forward-bar scheduler interval must be between 30 and 3600 seconds.")
        self._factory = factory
        self._interval_seconds = interval

    @property
    def interval_seconds(self) -> int:
        return self._interval_seconds

    async def run_forever(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self._factory.materialize_all)
            except Exception:
                logger.exception("BullionVault forward multi-horizon materialization failed.")
            await asyncio.sleep(self._interval_seconds)
