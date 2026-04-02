import logging

from pydantic import ValidationError

from firce.pipelines.simulation_pipeline import run_simulation_pipeline
from firce.pipelines.streaming_pipeline import run_streaming_pipeline
from firce.utils.arg_parser import parse_sim_args
from firce.utils.config import MonitorType, SimulationConfig
from firce.utils.logger import configure_sim_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Parse CLI arguments and launch the selected pipeline.
    """
    args = parse_sim_args()

    try:
        config = _build_config_from_args(args)
    except ValidationError as exc:
        logger.error('%s', exc)
        raise

    configure_sim_logging(config)
    logger.info('Simulation configuration: %s', config)
    _run_pipeline(config, args)


def _build_config_from_args(args) -> SimulationConfig:
    """
    Build the simulation configuration from parsed arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Simulation configuration.
    """
    return SimulationConfig(
        model_type=args.modelType,
        model_variant=args.modelVariant,
        ce_type=args.ceType,
        aggregated_path=args.aggregated_file,
        flows_path=args.flows_file,
        ce_kwargs={'folds': 5, 'significance': 0.05, 'random_state': args.seed},
        chunk_size=args.chunk_size,
        use_pca=args.use_pca,
        use_ASC=args.useASC,
        use_circular_logger=args.useCircularLogger,
        debug=args.debug,
        log_to_file=args.log2File,
        max_rows=args.max_rows,
        use_svm=args.useSVM,
        use_adaptive_chunking=args.useAC,
        is_unsw=args.unsw,
        use_mlp=args.useMLP,
        seed=args.seed,
        runNum=args.runNum,
        monitor_type=args.monitorType,
        monitor_kwargs=(
            {
                'dims': args.cadeDims,
                'margin': args.cadeMargin,
                'mad_threshold': args.cadeMadThreshold,
                'min_drift_ratio': args.cadeMinDriftRatio,
                'min_drift_count': args.cadeMinDriftCount,
                'batch_size': args.cadeBatchSize,
                'epochs': args.cadeEpochs,
                'lr': args.cadeLr,
                'cae_lambda_1': args.cadeLambda1,
                'similar_ratio': args.cadeSimilarRatio,
                'display_interval': args.cadeDisplayInterval,
                'force_retrain': args.cadeForceRetrain,
                'weights_path': args.cadeWeightsPath,
                'device': args.cadeDevice,
            }
            if args.monitorType == MonitorType.CADE
            else {}
        ),
    )


def _run_pipeline(config: SimulationConfig, args) -> None:
    """
    Run the selected pipeline.

    Args:
        config: Simulation configuration.
        args: Parsed CLI arguments.
    """
    pipeline = getattr(args, 'pipeline', 'simulation')
    if pipeline == 'simulation':
        run_simulation_pipeline(config)
        return

    if pipeline == 'streaming':
        run_streaming_pipeline(config)
        return

    raise ValueError(f'Unsupported pipeline: {pipeline}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        logger.exception('Fatal error during pipeline execution: %s', exc)
        raise
