# src/firce/pipelines/streaming_pipeline.py
"""
Live streaming pipeline for CE-based drift detection.

This pipeline bootstraps exactly like the offline simulation pipeline using
an initial aggregated dataset, then listens for live CSV batches and processes
them through the same chunk-processing, drift-detection, and retraining flow.
"""

from __future__ import annotations

import logging
import time

from typing import TYPE_CHECKING

from firce.runtime.bootstrap import SimulationRuntime, initialize_simulation_runtime
from firce.runtime.inference import process_chunk
from firce.utils.listener import StreamingBatchServer
from firce.utils.plotter import plot_results

if TYPE_CHECKING:
    from firce.utils.config import SimulationConfig

logger = logging.getLogger(__name__)


def run_streaming_pipeline(config: SimulationConfig) -> None:
    """
    Run the live streaming pipeline.

    This function initializes the runtime from an aggregated dataset, starts
    a local HTTP listener for incoming CSV batches, and processes each live
    batch through the same chunk-processing path used by the offline
    simulation pipeline.

    Args:
        config: Simulation configuration.
    """
    overall = time.perf_counter()
    runtime = initialize_simulation_runtime(config)

    server = StreamingBatchServer(
        host=getattr(config, 'listener_host', '127.0.0.1'),
        port=getattr(config, 'listener_port', 2048),
    )

    logger.info(
        'Starting live streaming pipeline with listener on %s:%d',
        server.host,
        server.port,
    )

    try:
        server.start()
        _run_live_loop(runtime, server)
    finally:
        server.stop()
        plot_results(config, overall, runtime.perf_stats)


def _run_live_loop(
    runtime: SimulationRuntime,
    server: StreamingBatchServer,
) -> None:
    """
    Process incoming live batches indefinitely.

    Args:
        runtime: Mutable simulation runtime.
        server: Running streaming batch server.
    """
    chunk_index = 0

    for batch in server.iter_batches():
        if batch.empty:
            logger.warning('Received empty live batch; skipping')
            continue

        logger.info(
            'Processing live batch %d with %d rows and %d columns',
            chunk_index,
            len(batch),
            len(batch.columns),
        )
        process_chunk(runtime, batch, chunk_index)
        chunk_index += 1
