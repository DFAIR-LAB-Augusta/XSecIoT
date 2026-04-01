import logging
import time

from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np

from core.config import SimulationConfig
from core.perf_stats import PerformanceStats

logger = logging.getLogger(__name__)


def graph_results(config: SimulationConfig, overall: float, perf_stats: PerformanceStats) -> None:
    """
    Log and visualize overall simulation metrics for CE-based drift detection.

    This function aggregates, logs, and saves plots for performance metrics gathered
    during conformal evaluation-based streaming simulation. It handles results including:
    - Total runtime
    - Accuracy trends over time (sliding average)
    - Drift detection frequency and intervals
    - CE and classifier performance metrics over retrainings
    - Chunk size variation over time (if adaptive chunking is enabled)

    Args:
        config (SimulationConfig): Configuration object containing CE, model, and simulation settings.
        overall (float): Time at the start of the simulation, used to compute total runtime.
        perf_stats (PerformanceStats): Object containing logs of all runtime and performance metrics,
            including accuracy logs, drift points, CE/classifier training scores, and chunk size history.

    Side Effects:
        - Saves multiple plots (accuracy, drift intervals, CE/classifier metrics, chunk size) to disk
          in the appropriate logging subdirectory.
        - Logs all summary statistics to the configured logging handler.
    """
    logger.info(f'[==OVERALL SIM STATS==] Total simulate time: {time.perf_counter() - overall:.4f}s')
    logger.info(f'Full performance stats: {perf_stats = }')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    if perf_stats.correct_log:
        final_accuracy = sum(perf_stats.correct_log) / len(perf_stats.correct_log)
        logger.info(f'[==OVERALL STATS==] Final Accuracy on all simulated samples: {final_accuracy:.4f}')

        window = 100
        moving_avg = np.convolve(perf_stats.correct_log, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(10, 4))
        plt.plot(moving_avg)
        plt.title('Sliding Accuracy Over Time')
        plt.xlabel('Flow Index')
        plt.ylabel('Accuracy (Window Size = 100)')
        plt.grid(True)
        plt.tight_layout()
        log_dir = Path('logging')
        log_dir.mkdir(exist_ok=True)
        if 'CETrain' in str(config.aggregated_path):
            ds_type = 'DFAIR'
        elif 'UNSW_NB15' in str(config.aggregated_path):
            ds_type = 'NB15'
        elif 'CIC_UNSW' in str(config.aggregated_path):
            ds_type = 'CIC_UNSW'
        else:
            raise ValueError('Expect dataset name not in aggregated_path')
        if config.use_adaptive_chunking:
            plot_path = (
                log_dir
                / 'ac'
                / ds_type
                / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_accuracy_plot.png'
            )
        else:
            plot_path = (
                log_dir
                / f'chunk_size_{config.chunk_size}'
                / ds_type
                / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_accuracy_plot.png'
            )
        plt.savefig(plot_path)
        logger.debug(f"Accuracy over time plot saved to '{plot_path}'")
        _summarize_timings('Per-Chunk Iteration Time', perf_stats.iteration_times)
        _summarize_timings('Per-Row Drift Detection Time', perf_stats.drift_times)

    if perf_stats.drift_detected_indices:
        total_drift = len(perf_stats.drift_detected_indices)
        drift_rate = total_drift / len(perf_stats.correct_log)

        logger.info(f'[==OVERALL SIM STATS==] Total Drift Detections: {total_drift}')
        logger.info(f'[==OVERALL SIM STATS==] Drift Detection Rate: {drift_rate:.4%}')

        if perf_stats.drift_intervals:
            avg_interval = np.mean(perf_stats.drift_intervals)
            logger.info(f'[==OVERALL SIM STATS==] Average Chunks Between Drift Detections: {avg_interval:.2f}')
            logger.info(f'[==OVERALL SIM STATS==] Drift Intervals (in chunks): {perf_stats.drift_intervals}')

            plt.figure(figsize=(10, 4))
            plt.plot(perf_stats.drift_intervals, marker='o')
            plt.title('Drift Intervals Over Time')
            plt.xlabel('Drift Detection Index')
            plt.ylabel(f'Chunks Since Last Drift (Chunk Size: {config.chunk_size})')
            plt.grid(True)
            log_dir = Path('logging')
            log_dir.mkdir(exist_ok=True)
            if 'CETrain' in str(config.aggregated_path):
                ds_type = 'DFAIR'
            elif 'UNSW_NB15' in str(config.aggregated_path):
                ds_type = 'NB15'
            elif 'CIC_UNSW' in str(config.aggregated_path):
                ds_type = 'CIC_UNSW'
            else:
                raise ValueError('Expect dataset name not in aggregated_path')
            if config.use_adaptive_chunking:
                plot_path = (
                    log_dir
                    / 'ac'
                    / ds_type
                    / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_drift_intervals.png'
                )
            else:
                plot_path = (
                    log_dir
                    / f'chunk_size_{config.chunk_size}'
                    / ds_type
                    / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_drift_intervals.png'
                )
            plt.savefig(plot_path)
            logger.debug(f"Drift interval plot saved to '{plot_path}'")

            plt.figure(figsize=(8, 4))
            plt.hist(perf_stats.drift_intervals, bins=range(1, max(perf_stats.drift_intervals) + 2), edgecolor='black')
            plt.title('Histogram of Drift Intervals (Chunks)')
            plt.xlabel('Chunks Between Drifts')
            plt.ylabel('Frequency')
            plt.grid(True)
            if config.use_adaptive_chunking:
                hist_path = (
                    log_dir
                    / 'ac'
                    / ds_type
                    / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_drift_interval_histogram.png'
                )
            else:
                hist_path = (
                    log_dir
                    / f'chunk_size_{config.chunk_size}'
                    / ds_type
                    / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_drift_interval_histogram.png'
                )
            plt.tight_layout()
            plt.savefig(hist_path)
            logger.debug(f"Drift interval histogram saved to '{hist_path}'")
    else:
        logger.info('[==OVERALL SIM STATS==] No Drift Detected')

    if perf_stats.ce_stats.accuracies:
        perf_stats.summarize_ce_metrics()

        _fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        metrics = [
            ('Accuracy', perf_stats.ce_stats.accuracies),
            ('Precision', perf_stats.ce_stats.precisions),
            ('Recall', perf_stats.ce_stats.recalls),
            ('F1 Score', perf_stats.ce_stats.f1s),
        ]
        for i, (title, data) in enumerate(metrics):
            axs[i].plot(data, marker='o')
            axs[i].set_title(f'CE {title} Over Calibrations')
            axs[i].set_xlabel('CE Calibration Index')
            axs[i].set_ylabel(title)
            axs[i].grid(True)

        plt.tight_layout()
        if config.use_adaptive_chunking:
            plot_dir = Path('logging') / 'ac'
        else:
            plot_dir = Path('logging') / f'chunk_size_{config.chunk_size}'
        plot_dir.mkdir(parents=True, exist_ok=True)
        if 'CETrain' in str(config.aggregated_path):
            ds_type = 'DFAIR'
        elif 'UNSW_NB15' in str(config.aggregated_path):
            ds_type = 'NB15'
        elif 'CIC_UNSW' in str(config.aggregated_path):
            ds_type = 'CIC_UNSW'
        else:
            raise ValueError('Expect dataset name not in aggregated_path')
        ce_metric_plot = (
            plot_dir
            / ds_type
            / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_ce_training_metrics.png'
        )
        plt.savefig(ce_metric_plot)
        logger.debug(f"CE training metric plot saved to '{ce_metric_plot}'")

    if perf_stats.classifier_stats.accuracies:
        perf_stats.summarize_classifier_metrics()

        _fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        metrics = [
            ('Accuracy', perf_stats.classifier_stats.accuracies),
            ('Precision', perf_stats.classifier_stats.precisions),
            ('Recall', perf_stats.classifier_stats.recalls),
            ('F1 Score', perf_stats.classifier_stats.f1s),
        ]
        for i, (title, data) in enumerate(metrics):
            axs[i].plot(data, marker='o')
            axs[i].set_title(f'Classifier {title} Over Calibrations')
            axs[i].set_xlabel('Classifier Calibration Index')
            axs[i].set_ylabel(title)
            axs[i].grid(True)

        plt.tight_layout()
        if config.use_adaptive_chunking:
            plot_dir = Path('logging') / 'ac'
        else:
            plot_dir = Path('logging') / f'chunk_size_{config.chunk_size}'
        plot_dir.mkdir(parents=True, exist_ok=True)
        if 'CETrain' in str(config.aggregated_path):
            ds_type = 'DFAIR'
        elif 'UNSW_NB15' in str(config.aggregated_path):
            ds_type = 'NB15'
        elif 'CIC_UNSW' in str(config.aggregated_path):
            ds_type = 'CIC_UNSW'
        else:
            raise ValueError('Expect dataset name not in aggregated_path')
        ce_metric_plot = (
            plot_dir
            / ds_type
            / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_classifier_training_metrics.png'
        )
        plt.savefig(ce_metric_plot)
        logger.debug(f"Classifier training metric plot saved to '{ce_metric_plot}'")

    if perf_stats.chunk_sizes and config.use_adaptive_chunking:
        logger.info(f'[==OVERALL SIM STATS==] Average Chunk Size: {mean(perf_stats.chunk_sizes):.2f}')
        logger.info(f'[==OVERALL SIM STATS==] Median Chunk Size: {median(perf_stats.chunk_sizes):.2f}')
        logger.info(f'[==OVERALL SIM STATS==] Standard Deviation of Chunk Sizes: {stdev(perf_stats.chunk_sizes):.2f}')
        plt.figure(figsize=(10, 4))
        plt.plot(perf_stats.chunk_sizes, marker='o')
        plt.title('Adaptive Chunk Size Over Time')
        plt.xlabel('Simulation Chunk Index')
        plt.ylabel('Chunk Size')
        plt.grid(True)
        plt.tight_layout()

        chunk_plot_dir = Path('logging') / 'ac'
        chunk_plot_dir.mkdir(parents=True, exist_ok=True)
        if 'CETrain' in str(config.aggregated_path):
            ds_type = 'DFAIR'
        elif 'UNSW_NB15' in str(config.aggregated_path):
            ds_type = 'NB15'
        elif 'CIC_UNSW' in str(config.aggregated_path):
            ds_type = 'CIC_UNSW'
        else:
            raise ValueError('Expect dataset name not in aggregated_path')
        chunk_plot_path = (
            chunk_plot_dir
            / ds_type
            / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_chunk_size_trace.png'
        )
        plt.savefig(chunk_plot_path)
        logger.debug(f"Chunk size over time plot saved to '{chunk_plot_path}'")


def _summarize_timings(name: str, times: list[float]) -> None:
    """
    Log summary statistics for a list of timing values.

    Computes and logs the count, mean, median, standard deviation,
    minimum, and maximum for the given list of timing durations (in seconds).
    If the list is empty, logs that no timings were recorded.

    Args:
        name (str): Descriptive label for the timing category (e.g., "Drift Detection Time").
        times (list[float]): List of timing values to summarize in seconds.
    """
    if not times:
        logger.info(f'{name}: No timings recorded.')
        return
    logger.info(
        f'[==OVERALL SIM STATS==] {name} — Count: {len(times)} | Mean: {mean(times):.4f}s | '
        f'Median: {median(times):.4f}s | Std: {stdev(times):.4f}s | '
        f'Min: {min(times):.4f}s | Max: {max(times):.4f}s'
    )
