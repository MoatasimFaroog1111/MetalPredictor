from __future__ import annotations

import asyncio
import logging

from metal_predictor.microstructure.collector import MicrostructureResearchCollector


logger = logging.getLogger(__name__)


class MicrostructureResearchScheduler:
    """Low-frequency collector kept far below BullionVault's view-market rate limit."""

    def __init__(
        self,
        collector: MicrostructureResearchCollector,
        *,
        interval_seconds: int = 60,
    ) -> None:
        interval = int(interval_seconds)
        if not 30 <= interval <= 3600:
            raise ValueError("Microstructure interval must be between 30 and 3600 seconds.")
        self._collector = collector
        self._interval_seconds = interval

    @property
    def interval_seconds(self) -> int:
        return self._interval_seconds

    async def run_forever(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self._collector.collect_once)
            except Exception:
                logger.exception("BullionVault microstructure research capture failed.")
            await asyncio.sleep(self._interval_seconds)
