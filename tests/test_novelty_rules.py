import sys

import numpy as np
import pytest
import torch

from pydantic import ValidationError

from firce.cli import _build_config_from_args
from firce.novelty.decision_rules import NOVELTY_RULES, apply_novelty_rule, is_novel
from firce.utils.arg_parser import parse_sim_args
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig


def _fixture(n=200, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    probas = rng.dirichlet(np.ones(n_classes), size=n)
    p_values = {c: rng.uniform(size=n) for c in range(n_classes)}
    return probas, p_values


def test_novelty_rule_names():
    assert NOVELTY_RULES == ('confidence', 'conformal', 'combined')


def test_novelty_rule_combined_equals_is_novel():
    probas, p_values = _fixture()
    expected = is_novel(probas, p_values, tau=0.6, alpha=0.3)
    actual = apply_novelty_rule('combined', probas, p_values, tau=0.6, alpha=0.3)
    assert actual.dtype == bool
    assert np.array_equal(actual, expected)


def test_novelty_rule_confidence_ignores_alpha():
    probas, p_values = _fixture()
    low = apply_novelty_rule('confidence', probas, p_values, tau=0.6, alpha=0.0)
    high = apply_novelty_rule('confidence', probas, p_values, tau=0.6, alpha=1.0)
    assert np.array_equal(low, high)
    assert np.array_equal(low, probas.max(axis=1) < 0.6)


def test_novelty_rule_conformal_ignores_tau():
    probas, p_values = _fixture()
    low = apply_novelty_rule('conformal', probas, p_values, tau=0.0, alpha=0.5)
    high = apply_novelty_rule('conformal', probas, p_values, tau=1.0, alpha=0.5)
    assert np.array_equal(low, high)
    matrix = np.stack(list(p_values.values()), axis=1)
    assert np.array_equal(low, np.all(matrix < 0.5, axis=1))


def test_novelty_rule_combined_is_intersection():
    probas, p_values = _fixture()
    conf = apply_novelty_rule('confidence', probas, p_values, tau=0.6, alpha=0.5)
    conf_only = apply_novelty_rule('conformal', probas, p_values, tau=0.6, alpha=0.5)
    both = apply_novelty_rule('combined', probas, p_values, tau=0.6, alpha=0.5)
    assert np.array_equal(both, conf & conf_only)


def test_novelty_rule_unknown_name_raises():
    probas, p_values = _fixture()
    with pytest.raises(ValueError, match='bogus'):
        apply_novelty_rule('bogus', probas, p_values, tau=0.6, alpha=0.3)


def test_novelty_rule_cli_flag_lands_on_config(monkeypatch, tmp_path):
    data = tmp_path / 'data.csv'
    data.write_text('a\n1\n')
    monkeypatch.setattr(sys, 'argv', ['prog', str(data), str(data), '--noveltyRule', 'conformal'])
    assert _build_config_from_args(parse_sim_args()).novelty_rule == 'conformal'
    monkeypatch.setattr(sys, 'argv', ['prog', str(data), str(data)])
    assert _build_config_from_args(parse_sim_args()).novelty_rule == 'combined'


def test_novelty_rule_invalid_name_fails_config_validation(tmp_path):
    data = tmp_path / 'data.csv'
    data.write_text('a\n1\n')
    with pytest.raises(ValidationError, match='novelty_rule'):
        SimulationConfig(
            model_type=ModelType.MULTI,
            model_variant=ModelVariant.DT,
            ce_type=CEType.NONE,
            aggregated_path=data,
            flows_path=data,
            device=torch.device('cpu'),
            novelty_rule='bogus',
        )
