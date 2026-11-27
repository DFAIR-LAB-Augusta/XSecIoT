import numpy as np
import pytest
import torch

from firce.experiments.driver import class_names_of
from firce.experiments.scoring import build_runtime, score_frame
from firce.experiments.split import split_device_time_blocks
from firce.models.mlp_ce import MLP_CE
from firce.models.mlp_ce_multiclass import MLPCEMulticlass
from firce.runtime.bootstrap import _build_monitor_model
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig
from tests.test_experiment_scoring import HELD_OUT, KNOWN, PRESET, _make_flows

CE_KWARGS = {'widths': (8,), 'epochs': 2, 'batch_size': 64}


@pytest.fixture(scope='module')
def flows():
    return _make_flows(n_per_device=200)


@pytest.fixture(scope='module')
def mlp_runtime(flows, tmp_path_factory):
    split = split_device_time_blocks(flows, PRESET, seed=0, block_minutes=120)
    runtime = build_runtime(
        split.train, PRESET, tmp_path_factory.mktemp('work'), 0, model_variant='feedforward', use_mlp=True
    )
    return runtime, split


def _csv(tmp_path):
    path = tmp_path / 'unused.csv'
    path.write_text('x\n')
    return path


def _config(model_type, path):
    return SimulationConfig(
        model_type=model_type,
        model_variant=ModelVariant.FEEDFORWARD,
        ce_type=CEType.ICE,
        monitor_type=MonitorType.CE,
        use_mlp=True,
        aggregated_path=path,
        flows_path=path,
        device=torch.device('cpu'),
    )


def test_multiclass_mlp_monitor_builds_multiclass_model(tmp_path):
    y_train = np.array(['b', 'a', 'c', 'a'])
    model = _build_monitor_model(_config(ModelType.MULTI, _csv(tmp_path)), None, 5, dict(CE_KWARGS), y_train=y_train)
    assert isinstance(model, MLPCEMulticlass)
    assert model.classes_.tolist() == ['a', 'b', 'c']


def test_multiclass_mlp_monitor_binary_still_builds_mlp_ce(tmp_path):
    model = _build_monitor_model(_config(ModelType.BINARY, _csv(tmp_path)), None, 5, dict(CE_KWARGS))
    assert isinstance(model, MLP_CE)


def test_multiclass_mlp_monitor_runtime_scores_known_classes(mlp_runtime):
    runtime, split = mlp_runtime
    assert isinstance(runtime.monitor.model, MLPCEMulticlass)
    scored = score_frame(runtime, split.val, PRESET)
    assert scored.probas.shape == (len(split.val), len(KNOWN))
    assert set(scored.p_values) == set(KNOWN)
    assert HELD_OUT not in scored.p_values


def test_multiclass_mlp_monitor_class_names_match_p_value_keys(mlp_runtime):
    runtime, split = mlp_runtime
    scored = score_frame(runtime, split.val, PRESET)
    assert set(class_names_of(runtime)) == set(scored.p_values)
