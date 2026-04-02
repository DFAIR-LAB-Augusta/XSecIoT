import logging

from pathlib import Path

from firce.utils.config import SimulationConfig


def configure_sim_logging(config: SimulationConfig) -> None:
    """
    Configure logging to console, and optionally to a log file.

    Args:
        config (SimulationConfig): Configuration object containing model/CE info.
    """
    log_level = logging.DEBUG if config.debug else logging.INFO
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]

    if config.log_to_file:
        log_dir = Path('logging')
        log_dir.mkdir(exist_ok=True)
        if config.use_adaptive_chunking:
            log_dir = log_dir / 'ac'
        else:
            log_dir = log_dir / f'chunk_size_{config.chunk_size}'
        log_dir.mkdir(exist_ok=True)
        if 'CETrain' in str(config.aggregated_path):
            ds_type = 'DFAIR'
        elif 'UNSW_NB15' in str(config.aggregated_path):
            ds_type = 'NB15'
        elif 'CIC_UNSW' in str(config.aggregated_path):
            ds_type = 'CIC_UNSW'
        else:
            raise ValueError('Expect dataset name not in aggregated_path')
        log_dir = log_dir / ds_type
        log_dir.mkdir(exist_ok=True)
        log_file = (
            log_dir
            / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_{config.seed}_{config.runNum}_run.log'
        )
        file_handler = logging.FileHandler(log_file, mode='w')
        log_handlers.append(file_handler)

    logging.basicConfig(
        level=log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s', handlers=log_handlers, force=True
    )
    logging.captureWarnings(True)


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
