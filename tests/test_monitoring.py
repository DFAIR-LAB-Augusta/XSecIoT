import pytest

from firce.runtime.monitoring import filter_ce_kwargs
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_filter_ce_kwargs_raises_when_ce_disabled(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.NONE)

    with pytest.raises(RuntimeError, match='CE is disabled'):
        filter_ce_kwargs(config)


def test_filter_ce_kwargs_keeps_only_constructor_params_for_ice(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.ICE, ce_kwargs={'calibration_split': 0.3, 'not_a_real_param': 123})

    result = filter_ce_kwargs(config)

    assert result == {'calibration_split': 0.3}


def test_filter_ce_kwargs_keeps_only_constructor_params_for_cce(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.CCE, ce_kwargs={'folds': 3, 'not_a_real_param': 123})

    result = filter_ce_kwargs(config)

    assert result == {'folds': 3}


def test_filter_ce_kwargs_empty_kwargs_returns_empty_dict(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.APPROX_TCE, ce_kwargs={})

    result = filter_ce_kwargs(config)

    assert result == {}
