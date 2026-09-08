import torch

from firce.runtime.bootstrap import get_rolling_columns
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig

DEVICE = torch.device('cpu')


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_get_rolling_columns_multiclass_includes_mc_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI)
    columns = get_rolling_columns(config)

    assert 'MC_Label' in columns
    assert 'BinLabel' not in columns


def test_get_rolling_columns_binary_still_includes_bin_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)
    columns = get_rolling_columns(config)

    assert 'BinLabel' in columns
    assert 'MC_Label' not in columns
