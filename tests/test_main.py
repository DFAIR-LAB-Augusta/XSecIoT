import sys

import pytest

import fire.main as fire_main

from fire.main import _parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv'])
    args = _parse_args()

    assert args.dataset_path == 'data.csv'
    assert args.window_size == '5s'
    assert args.step_size == '1s'
    assert not args.unsw
    assert not args.pca
    assert not args.noPre
    assert not args.noMod
    assert not args.noSim


def test_parse_args_skip_flags(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv', '--noPre', '--noMod', '--noSim', '--unsw'])
    args = _parse_args()

    assert args.noPre
    assert args.noMod
    assert args.noSim
    assert args.unsw


def _stub_leaf_functions(monkeypatch, calls):
    monkeypatch.setattr(fire_main, 'run_preprocessing', lambda *a, **kw: calls.append(('run_preprocessing', a, kw)))
    monkeypatch.setattr(
        fire_main, 'run_binary_classification', lambda *a, **kw: calls.append(('run_binary_classification', a, kw))
    )
    monkeypatch.setattr(
        fire_main,
        'run_multiclass_classification',
        lambda *a, **kw: calls.append(('run_multiclass_classification', a, kw)),
    )
    monkeypatch.setattr(
        fire_main, 'run_feature_engineering', lambda *a, **kw: calls.append(('run_feature_engineering', a, kw))
    )
    monkeypatch.setattr(
        fire_main, 'sequential_simulation', lambda *a, **kw: calls.append(('sequential_simulation', a, kw)) or []
    )
    monkeypatch.setattr(
        fire_main,
        'continuous_simulation',
        lambda *a, **kw: calls.append(('continuous_simulation', a, kw)) or (None, []),
    )
    monkeypatch.setattr(
        fire_main, 'parallel_simulation', lambda *a, **kw: calls.append(('parallel_simulation', a, kw)) or []
    )


def test_main_no_pre_skips_preprocessing(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noMod', '--noSim'])

    fire_main.main()

    assert calls == []


def test_main_no_mod_skips_modeling(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noMod', '--noSim'])

    fire_main.main()

    names = [c[0] for c in calls]
    assert names == ['run_preprocessing']


def test_main_no_sim_skips_simulations(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noSim'])

    fire_main.main()

    names = [c[0] for c in calls]
    assert names == ['run_binary_classification', 'run_multiclass_classification', 'run_feature_engineering']


def test_main_no_pre_without_aggregated_file_exits(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre'])

    with pytest.raises(SystemExit) as exc_info:
        fire_main.main()

    assert exc_info.value.code == 1
    assert calls == []


def test_main_modeling_stage_threads_unsw_and_pca_flags(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noSim', '--unsw', '--pca'])

    fire_main.main()

    call_map = {name: (args, kwargs) for name, args, kwargs in calls}
    assert call_map['run_binary_classification'][0] == (str(aggregated), True, True)
    assert call_map['run_multiclass_classification'][0] == (str(aggregated), True, True)
    assert call_map['run_feature_engineering'][0] == (str(aggregated),)


def test_main_simulation_stage_covers_full_sweep(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noMod'])

    fire_main.main()

    sim_calls = [c for c in calls if c[0] in ('sequential_simulation', 'continuous_simulation', 'parallel_simulation')]
    assert len(sim_calls) == 30

    seen_combos = set()
    for name, _args, kwargs in sim_calls:
        assert kwargs['aggregated_file'] == str(aggregated)
        assert kwargs['threshold'] == 0.5
        assert kwargs['isUNSW'] is False
        seen_combos.add((kwargs['model_type'], kwargs['model_variant'], name))

    expected_modes = {
        'sequential': 'sequential_simulation',
        'continuous': 'continuous_simulation',
        'parallel': 'parallel_simulation',
    }
    expected_combos = {
        (model_type, variant, expected_modes[mode])
        for model_type in ('binary', 'multi')
        for variant in ('dt', 'knn', 'rf', 'feedforward', 'xgb')
        for mode in ('sequential', 'continuous', 'parallel')
    }
    assert seen_combos == expected_combos
