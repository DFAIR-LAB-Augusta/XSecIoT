from __future__ import annotations

from typing import Any

from src.core.config import MonitorType, SimulationConfig
from src.core.drift_monitor.conformal_monitor import ConformalDriftMonitor


def build_monitor(
    config: SimulationConfig,
    model: Any,
    significance_controller: Any = None,
):
    """Create the configured drift monitor backend."""
    if config.monitor_type == MonitorType.CE:
        return ConformalDriftMonitor(
            ce_type=config.ce_type,
            model=model,
            significance_controller=significance_controller,
            **config.ce_kwargs,
        )

    if config.monitor_type == MonitorType.NONE:
        return None

    if config.monitor_type == MonitorType.CADE:
        from src.core.drift_monitor.cade_monitor import CadeDriftMonitor

        return CadeDriftMonitor(
            config=config,
        )

    raise ValueError(f"Unsupported monitor_type: {config.monitor_type}")
