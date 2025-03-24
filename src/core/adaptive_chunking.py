# src/xseciot_core/adaptive_chunker

"""
Adaptive Chunk Controller

This module defines an AdaptiveChunkController class that adjusts the size
of streaming data chunks dynamically based on drift detection feedback.
It uses an exponential moving average (EMA) of the drift rate and a cooldown
mechanism to prevent rapid oscillation in chunk size.

Typical usage:
    >>> controller = AdaptiveChunkController(init_chunk_size=1000)
    >>> for chunk in chunk_stream:
    >>>     ...
    >>>     controller.update(drift_detected=True)
    >>>     new_chunk_size = controller.get_chunk_size()
"""

import logging

from src.core.config import AdaptiveChunkConfig
from src.core.perf_stats import PerformanceStats

logger = logging.getLogger(__name__)


class AdaptiveChunkController:
    """
    Controller that adaptively adjusts chunk size based on drift frequency using EMA.

    Attributes:
        chunk_size (int): Current chunk size.
        min_chunk_size (int): Lower bound on chunk size.
        max_chunk_size (int): Upper bound on chunk size.
        ema_decay (float): Exponential decay factor for smoothing drift rate.
        cooldown_period (int): Number of updates to wait before adjusting size again.
    """
    def __init__(
        self,
        ac_config: AdaptiveChunkConfig,
    ) -> None:
        self.chunk_size: int = ac_config.init_chunk_size
        self.min_chunk_size: int = ac_config.min_chunk_size
        self.max_chunk_size: int = ac_config.max_chunk_size
        self.ema_decay: float = ac_config.ema_decay
        self.cooldown_period: int = ac_config.cooldown_period
        self.step_size: int = ac_config.step_size

        self._cooldown_counter: int = 0
        self._drift_rate_ema: float = 0.0
        self._total_chunks: int = 0
        self._total_drifts: int = 0

    def update(self, drift_detected: bool, perf_stats: PerformanceStats) -> None:
        """
        Update internal state with the result of the current chunk's drift detection.

        Args:
            drift_detected (bool): Whether drift was detected in the current chunk.
        """
        self._total_chunks += 1
        if drift_detected:
            self._total_drifts += 1

        raw_drift_rate = self._total_drifts / self._total_chunks
        self._drift_rate_ema = (
            self.ema_decay * self._drift_rate_ema
            + (1 - self.ema_decay) * raw_drift_rate
        )

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        self._adjust_chunk_size()
        self._cooldown_counter = self.cooldown_period
        perf_stats.chunk_sizes.append(self.chunk_size)

    def _adjust_chunk_size(self) -> None:
        """
        Adjust the chunk size based on the current EMA of the drift rate.
        """
        previous_size = self.chunk_size
        if self._drift_rate_ema > 0.2:
            self.chunk_size = max(self.min_chunk_size, self.chunk_size // 2)
        elif self._drift_rate_ema < 0.05:
            self.chunk_size = min(self.max_chunk_size, self.chunk_size + self.step_size)

        if self.chunk_size != previous_size:
            logger.info(
                f"[AdaptiveChunking] Chunk size changed from {previous_size} to {self.chunk_size} "
                f"(drift EMA: {self._drift_rate_ema:.4f})"
            )
        else:
            logger.debug(
                f"[AdaptiveChunking] No change in chunk size (current: {self.chunk_size}, "
                f"drift EMA: {self._drift_rate_ema:.4f})"
            )

    def get_chunk_size(self) -> int:
        """
        Get the current chunk size.

        Returns:
            int: The current chunk size to use for streaming.
        """
        return self.chunk_size

    def reset(self) -> None:
        """
        Reset all internal counters and metrics to initial state.
        """
        self._cooldown_counter = 0
        self._drift_rate_ema = 0.0
        self._total_chunks = 0
        self._total_drifts = 0
