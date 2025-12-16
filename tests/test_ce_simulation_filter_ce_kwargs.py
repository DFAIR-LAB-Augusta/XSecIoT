import inspect
import os

from types import SimpleNamespace

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


@pytest.mark.parametrize(
    'ce_type, impl_cls',
    [
        (sim.CEType.ICE, sim.InductiveConformalEvaluator),
        (sim.CEType.CCE, sim.CrossConformalEvaluator),
        (sim.CEType.APPROX_TCE, sim.ApproximateTransductiveConformalEvaluator),
        (sim.CEType.APPROX_CCE, sim.ApproxCrossConformalEvaluator),
    ],
)
def test_filter_ce_kwargs_filters_to_constructor_signature(ce_type, impl_cls) -> None:
    sig = inspect.signature(impl_cls.__init__)
    supplied = {
        'random_state': 123,
        'calibration_split': 0.2,
        'significance': 0.15,
        'this_is_not_real': 'nope',
    }

    config = SimpleNamespace(ce_type=ce_type, ce_kwargs=supplied)

    out = sim._filter_ce_kwargs(config)  # type: ignore
    assert 'this_is_not_real' not in out
    assert set(out.keys()).issubset(set(sig.parameters.keys()))
    # sanity: at least one common kw should survive for most evaluators
    assert any(k in out for k in ('random_state', 'calibration_split', 'significance'))


def test_filter_ce_kwargs_raises_when_none() -> None:
    config = SimpleNamespace(ce_type=sim.CEType.NONE, ce_kwargs={'random_state': 1})
    with pytest.raises(RuntimeError):
        sim._filter_ce_kwargs(config)  # type: ignore
