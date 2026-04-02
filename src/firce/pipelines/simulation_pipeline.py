import logging
import time

import pandas as pd

from firce.adaptive_chunking import AdaptiveChunkController
from firce.runtime.bootstrap import SimulationRuntime, initialize_simulation_runtime
from firce.runtime.inference import process_chunk
from firce.utils.plotter import plot_results

logger = logging.getLogger(__name__)


def run_simulation_pipeline(config) -> None:
    """
    Run the offline simulation pipeline.

    Args:
        config: Simulation configuration.
    """
    overall = time.perf_counter()
    runtime = initialize_simulation_runtime(config)

    try:
        if config.use_adaptive_chunking:
            _run_adaptive_chunk_loop(runtime)
        else:
            _run_fixed_chunk_loop(runtime)
    finally:
        plot_results(config, overall, runtime.perf_stats)


def _run_fixed_chunk_loop(runtime: SimulationRuntime) -> None:
    """
    Process flows using a fixed chunk size.

    Args:
        runtime: Mutable simulation runtime.
    """
    for chunk_num, chunk in enumerate(
        pd.read_csv(
            runtime.config.flows_path,
            chunksize=runtime.config.chunk_size,
        )
    ):
        process_chunk(runtime, chunk, chunk_num)


def _run_adaptive_chunk_loop(runtime: SimulationRuntime) -> None:
    """
    Process flows using adaptive chunk sizing.

    Args:
        runtime: Mutable simulation runtime.
    """
    chunker = AdaptiveChunkController(runtime.config.adaptive_chunk_config)
    logger.info(
        "[AdaptiveChunking] Enabled. Initial chunk size: %d",
        chunker.get_chunk_size(),
    )

    flow_data = pd.read_csv(
        runtime.config.flows_path,
        iterator=True,
        chunksize=1_000_000,
    )
    current_df = pd.DataFrame()
    chunk_index = 0

    for big_batch in flow_data:
        current_df = pd.concat([current_df, big_batch], ignore_index=True)

        while len(current_df) >= chunker.get_chunk_size():
            chunk_size = chunker.get_chunk_size()
            chunk = current_df.iloc[:chunk_size]
            current_df = current_df.iloc[chunk_size:]

            process_chunk(runtime, chunk, chunk_index)

            drift_occurred = (
                len(runtime.perf_stats.drift_detected_indices) > 0
                and runtime.perf_stats.drift_detected_indices[-1] == chunk_index
            )
            chunker.update(drift_occurred, runtime.perf_stats)
            chunk_index += 1