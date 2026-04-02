import logging
import time

from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np

from firce.utils.config import SimulationConfig
from firce.utils.perf_stats import PerformanceStats

logger = logging.getLogger(__name__)


def plot_results(
    config: SimulationConfig,
    overall: float,
    perf_stats: PerformanceStats,
) -> None:
    """
    Log and visualize overall simulation metrics for CE-based drift detection.

    This function aggregates, logs, and saves plots for performance metrics
    gathered during conformal evaluation-based streaming simulation.

    Args:
        config: Configuration object containing CE, model, and simulation
            settings.
        overall: Time at the start of the simulation, used to compute total
            runtime.
        perf_stats: Object containing logs of runtime and performance metrics,
            including accuracy logs, drift points, CE/classifier training
            scores, and chunk size history.

    Side Effects:
        - Saves multiple plots to disk.
        - Logs summary statistics to the configured logging handler.
    """
    _log_overall_runtime(overall, perf_stats)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    if perf_stats.correct_log:
        _handle_accuracy_results(config, perf_stats)

    if perf_stats.drift_detected_indices:
        _handle_drift_results(config, perf_stats)
    else:
        logger.info('[==OVERALL SIM STATS==] No Drift Detected')

    if perf_stats.ce_stats.accuracies:
        perf_stats.summarize_ce_metrics()
        _plot_metric_grid(
            config=config,
            metric_prefix='CE',
            metrics=[
                ('Accuracy', perf_stats.ce_stats.accuracies),
                ('Precision', perf_stats.ce_stats.precisions),
                ('Recall', perf_stats.ce_stats.recalls),
                ('F1 Score', perf_stats.ce_stats.f1s),
            ],
            x_label='CE Calibration Index',
            output_suffix='ce_training_metrics.png',
        )

    if perf_stats.classifier_stats.accuracies:
        perf_stats.summarize_classifier_metrics()
        _plot_metric_grid(
            config=config,
            metric_prefix='Classifier',
            metrics=[
                ('Accuracy', perf_stats.classifier_stats.accuracies),
                ('Precision', perf_stats.classifier_stats.precisions),
                ('Recall', perf_stats.classifier_stats.recalls),
                ('F1 Score', perf_stats.classifier_stats.f1s),
            ],
            x_label='Classifier Calibration Index',
            output_suffix='classifier_training_metrics.png',
        )

    if perf_stats.chunk_sizes and config.use_adaptive_chunking:
        _handle_chunk_size_results(config, perf_stats)


def _log_overall_runtime(overall: float, perf_stats: PerformanceStats) -> None:
    """
    Log top-level runtime and performance object details.

    Args:
        overall: Simulation start time from ``time.perf_counter()``.
        perf_stats: Performance statistics collected during simulation.
    """
    total_time = time.perf_counter() - overall
    logger.info(f'[==OVERALL SIM STATS==] Total simulate time: {total_time:.4f}s')
    logger.info(f'Full performance stats: {perf_stats = }')


def _handle_accuracy_results(
    config: SimulationConfig,
    perf_stats: PerformanceStats,
) -> None:
    """
    Log final accuracy, save sliding-accuracy plot, and summarize timings.

    Args:
        config: Simulation configuration.
        perf_stats: Performance statistics collected during simulation.
    """
    final_accuracy = sum(perf_stats.correct_log) / len(perf_stats.correct_log)
    logger.info(f'[==OVERALL STATS==] Final Accuracy on all simulated samples: {final_accuracy:.4f}')

    window = 100
    moving_avg = np.convolve(
        perf_stats.correct_log,
        np.ones(window) / window,
        mode='valid',
    )

    plt.figure(figsize=(10, 4))
    plt.plot(moving_avg)
    plt.title('Sliding Accuracy Over Time')
    plt.xlabel('Flow Index')
    plt.ylabel('Accuracy (Window Size = 100)')
    plt.grid(True)
    plt.tight_layout()

    plot_path = _build_plot_path(config, 'accuracy_plot.png')
    _save_current_figure(plot_path, 'Accuracy over time plot')

    _summarize_timings('Per-Chunk Iteration Time', perf_stats.iteration_times)
    _summarize_timings('Per-Row Drift Detection Time', perf_stats.drift_times)


def _handle_drift_results(
    config: SimulationConfig,
    perf_stats: PerformanceStats,
) -> None:
    """
    Log drift summary statistics and save drift-related plots.

    Args:
        config: Simulation configuration.
        perf_stats: Performance statistics collected during simulation.
    """
    total_drift = len(perf_stats.drift_detected_indices)
    drift_rate = total_drift / len(perf_stats.correct_log)

    logger.info(f'[==OVERALL SIM STATS==] Total Drift Detections: {total_drift}')
    logger.info(f'[==OVERALL SIM STATS==] Drift Detection Rate: {drift_rate:.4%}')

    if not perf_stats.drift_intervals:
        return

    avg_interval = np.mean(perf_stats.drift_intervals)
    logger.info(f'[==OVERALL SIM STATS==] Average Chunks Between Drift Detections: {avg_interval:.2f}')
    logger.info(f'[==OVERALL SIM STATS==] Drift Intervals (in chunks): {perf_stats.drift_intervals}')

    _plot_drift_intervals(config, perf_stats.drift_intervals, config.chunk_size)
    _plot_drift_interval_histogram(config, perf_stats.drift_intervals)


def _handle_chunk_size_results(
    config: SimulationConfig,
    perf_stats: PerformanceStats,
) -> None:
    """
    Log adaptive chunk size statistics and save a chunk-size trace plot.

    Args:
        config: Simulation configuration.
        perf_stats: Performance statistics collected during simulation.
    """
    logger.info(f'[==OVERALL SIM STATS==] Average Chunk Size: {mean(perf_stats.chunk_sizes):.2f}')
    logger.info(f'[==OVERALL SIM STATS==] Median Chunk Size: {median(perf_stats.chunk_sizes):.2f}')

    if len(perf_stats.chunk_sizes) > 1:
        chunk_std = stdev(perf_stats.chunk_sizes)
    else:
        chunk_std = 0.0

    logger.info(f'[==OVERALL SIM STATS==] Standard Deviation of Chunk Sizes: {chunk_std:.2f}')

    plt.figure(figsize=(10, 4))
    plt.plot(perf_stats.chunk_sizes, marker='o')
    plt.title('Adaptive Chunk Size Over Time')
    plt.xlabel('Simulation Chunk Index')
    plt.ylabel('Chunk Size')
    plt.grid(True)
    plt.tight_layout()

    plot_path = _build_plot_path(config, 'chunk_size_trace.png')
    _save_current_figure(plot_path, 'Chunk size over time plot')


def _plot_drift_intervals(
    config: SimulationConfig,
    drift_intervals: list[int],
    chunk_size: int,
) -> None:
    """
    Plot and save drift intervals over time.

    Args:
        config: Simulation configuration.
        drift_intervals: Number of chunks between successive drifts.
        chunk_size: Current configured chunk size.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(drift_intervals, marker='o')
    plt.title('Drift Intervals Over Time')
    plt.xlabel('Drift Detection Index')
    plt.ylabel(f'Chunks Since Last Drift (Chunk Size: {chunk_size})')
    plt.grid(True)
    plt.tight_layout()

    plot_path = _build_plot_path(config, 'drift_intervals.png')
    _save_current_figure(plot_path, 'Drift interval plot')


def _plot_drift_interval_histogram(
    config: SimulationConfig,
    drift_intervals: list[int],
) -> None:
    """
    Plot and save a histogram of drift intervals.

    Args:
        config: Simulation configuration.
        drift_intervals: Number of chunks between successive drifts.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(
        drift_intervals,
        bins=range(1, max(drift_intervals) + 2),
        edgecolor='black',
    )
    plt.title('Histogram of Drift Intervals (Chunks)')
    plt.xlabel('Chunks Between Drifts')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    plot_path = _build_plot_path(config, 'drift_interval_histogram.png')
    _save_current_figure(plot_path, 'Drift interval histogram')


def _plot_metric_grid(
    config: SimulationConfig,
    metric_prefix: str,
    metrics: list[tuple[str, list[float]]],
    x_label: str,
    output_suffix: str,
) -> None:
    """
    Plot a reusable 2x2 metric grid and save it.

    Args:
        config: Simulation configuration.
        metric_prefix: Prefix used in subplot titles, such as ``CE`` or
            ``Classifier``.
        metrics: Metric title and series pairs to plot.
        x_label: Label for the x-axis on each subplot.
        output_suffix: Filename suffix for the saved plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axs.flatten()

    for axis, (title, data) in zip(flat_axes, metrics):
        axis.plot(data, marker='o')
        axis.set_title(f'{metric_prefix} {title} Over Calibrations')
        axis.set_xlabel(x_label)
        axis.set_ylabel(title)
        axis.grid(True)

    fig.tight_layout()

    plot_path = _build_plot_path(config, output_suffix)
    _save_figure(fig, plot_path, f'{metric_prefix} training metric plot')


def _build_plot_path(config: SimulationConfig, suffix: str) -> Path:
    """
    Build the output path for a plot based on simulation configuration.

    Args:
        config: Simulation configuration.
        suffix: Filename suffix for the output plot.

    Returns:
        Fully resolved output path for the plot file.

    Raises:
        ValueError: If the dataset name cannot be inferred from
            ``config.aggregated_path``.
    """
    ds_type = _resolve_dataset_type(config)

    if config.use_adaptive_chunking:
        base_dir = Path('logging') / 'ac' / ds_type
    else:
        base_dir = Path('logging') / f'chunk_size_{config.chunk_size}' / ds_type

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / _build_plot_filename(config, suffix)


def _build_plot_filename(config: SimulationConfig, suffix: str) -> str:
    """
    Build a standardized plot filename.

    Args:
        config: Simulation configuration.
        suffix: Filename suffix for the output plot.

    Returns:
        Standardized plot filename.
    """
    return (
        f'{config.model_variant.value}_'
        f'{config.ce_type.value}_'
        f'{config.model_type.value}_'
        f'{config.seed}_'
        f'{config.runNum}_'
        f'{suffix}'
    )


def _resolve_dataset_type(config: SimulationConfig) -> str:
    """
    Infer the dataset label from the aggregated input path.

    Args:
        config: Simulation configuration.

    Returns:
        Dataset label used in logging directory structure.

    Raises:
        ValueError: If the dataset name cannot be inferred.
    """
    aggregated_path = str(config.aggregated_path)

    if 'CETrain' in aggregated_path:
        return 'DFAIR'
    if 'UNSW_NB15' in aggregated_path:
        return 'NB15'
    if 'CIC_UNSW' in aggregated_path:
        return 'CIC_UNSW'

    raise ValueError('Expected dataset name not found in aggregated_path')


def _save_current_figure(plot_path: Path, description: str) -> None:
    """
    Save the current Matplotlib figure and close it.

    Args:
        plot_path: Destination path for the figure.
        description: Human-readable plot description for debug logging.
    """
    plt.savefig(plot_path)
    plt.close()
    logger.debug(f"{description} saved to '{plot_path}'")


def _save_figure(fig: plt.Figure, plot_path: Path, description: str) -> None:  # pyright: ignore[reportPrivateImportUsage]
    """
    Save a specific Matplotlib figure and close it.

    Args:
        fig: Figure object to save.
        plot_path: Destination path for the figure.
        description: Human-readable plot description for debug logging.
    """
    fig.savefig(plot_path)
    plt.close(fig)
    logger.debug(f"{description} saved to '{plot_path}'")


def _summarize_timings(name: str, times: list[float]) -> None:
    """
    Log summary statistics for a list of timing values.

    Computes and logs the count, mean, median, standard deviation, minimum,
    and maximum for the given list of timing durations in seconds. If the
    list is empty, logs that no timings were recorded.

    Args:
        name: Descriptive label for the timing category.
        times: Timing values to summarize in seconds.
    """
    if not times:
        logger.info(f'{name}: No timings recorded.')
        return

    std_value = stdev(times) if len(times) > 1 else 0.0
    logger.info(
        f'[==OVERALL SIM STATS==] {name} | Count: {len(times)} | '
        f'Mean: {mean(times):.4f}s | Median: {median(times):.4f}s | '
        f'Std: {std_value:.4f}s | Min: {min(times):.4f}s | '
        f'Max: {max(times):.4f}s'
    )


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly.')
