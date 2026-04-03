import inspect

from typing import Any

from firce.conformalEval.approx_cce import ApproxCrossConformalEvaluator
from firce.conformalEval.cce import CrossConformalEvaluator
from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.conformalEval.tce import ApproximateTransductiveConformalEvaluator
from firce.utils.config import CEType, SimulationConfig


def filter_ce_kwargs(config: SimulationConfig) -> dict[str, Any]:
    """
    Filter CE kwargs to those accepted by the selected evaluator type.

    Args:
        config: Simulation configuration.

    Returns:
        Filtered CE constructor kwargs.

    Raises:
        RuntimeError: If CE is disabled.
    """
    if config.ce_type == CEType.NONE:
        raise RuntimeError('CE is disabled; CE kwargs were requested unexpectedly.')

    impl_map = {
        'ice': InductiveConformalEvaluator,
        'cce': CrossConformalEvaluator,
        'approx_tce': ApproximateTransductiveConformalEvaluator,
        'approx_cce': ApproxCrossConformalEvaluator,
    }
    impl_cls = impl_map[config.ce_type.value]
    signature = inspect.signature(impl_cls.__init__)
    return {key: value for key, value in config.ce_kwargs.items() if key in signature.parameters}
