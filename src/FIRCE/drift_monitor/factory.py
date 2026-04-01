from __future__ import annotations

import logging

from typing import Any

from FIRCE.config import MonitorType, SimulationConfig
from FIRCE.drift_monitor.conformal_monitor import ConformalDriftMonitor

logger = logging.getLogger(__name__)


def build_monitor(
    config: SimulationConfig,
    model: Any,
    significance_controller: Any = None,
):
    """Create the configured drift monitor backend."""
    if config.monitor_type == MonitorType.CE:
        logger.info('Using CE drift monitor')
        return ConformalDriftMonitor(
            ce_type=config.ce_type,
            model=model,
            significance_controller=significance_controller,
            **config.ce_kwargs,
        )

    if config.monitor_type == MonitorType.NONE:
        logger.info('No drift monitor used')
        return None

    if config.monitor_type == MonitorType.CADE:
        from FIRCE.drift_monitor.cade_monitor import CadeDriftMonitor

        logger.info('Using CADE drift monitor')
        return CadeDriftMonitor(
            config=config,
        )

    raise ValueError(f'Unsupported monitor_type: {config.monitor_type}')
