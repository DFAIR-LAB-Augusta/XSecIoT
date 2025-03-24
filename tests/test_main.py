import importlib
import sys

from pathlib import Path

import pytest

# Make sure our `src/` folder is on PYTHONPATH so import FIRE.main
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import src.FIRE.main as main_mod  # noqa: E402


def make_dataset(tmp_path):
    """
    Create tmp_path/DS/raw.csv and tmp_path/DS/aggregated_data.csv
    so main.py's logic finds both.
    """
    ds = tmp_path / "DS"
    ds.mkdir()
    raw = ds / "raw.csv"
    raw.write_text("a,b\n1,2\n")
    agg = ds / "aggregated_data.csv"
    agg.write_text("a,b\n3,4\n")
    return str(raw), str(agg)


def test_all_steps_are_called(tmp_path, monkeypatch):
    raw, agg = make_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    importlib.reload(main_mod)
    calls = {'pre': 0, 'bin': 0, 'multi': 0, 'fe': 0, 'seq': 0, 'cont': 0, 'par': 0}

    # stub out each pipeline function
    monkeypatch.setattr(main_mod, 'run_preprocessing',
        lambda dataset_path, window_size, step_size, unsw: calls.__setitem__('pre', calls['pre'] + 1))
    monkeypatch.setattr(main_mod, 'run_binary_classification',
        lambda agg_f, unsw, pca: calls.__setitem__('bin', calls['bin'] + 1))
    monkeypatch.setattr(main_mod, 'run_multiclass_classification',
        lambda agg_f, unsw, pca: calls.__setitem__('multi', calls['multi'] + 1))
    monkeypatch.setattr(main_mod, 'run_feature_engineering',
        lambda agg_f: calls.__setitem__('fe', calls['fe'] + 1))
    # for simulations, return a list of length 2 so main can call len()
    monkeypatch.setattr(main_mod, 'sequential_simulation',
        lambda **kwargs: calls.__setitem__('seq', calls['seq'] + 1) or [0, 1])
    monkeypatch.setattr(main_mod, 'continuous_simulation',
        lambda **kwargs: ([], calls.__setitem__('cont', calls['cont'] + 1) or [0, 1]))
    monkeypatch.setattr(main_mod, 'parallel_simulation',
        lambda **kwargs: calls.__setitem__('par', calls['par'] + 1) or [0, 1])

    # Run with no flags
    sys.argv = ["prog", raw]
    main_mod.main()

    # All preprocessing & modeling steps should run exactly once
    assert calls['pre'] == 1
    assert calls['bin'] == 1
    assert calls['multi'] == 1
    assert calls['fe'] == 1

    # Simulation calls: 2 model_types * 5 variants each per mode = 10 calls per mode
    assert calls['seq'] == 2 * 5
    assert calls['cont'] == 2 * 5
    assert calls['par'] == 2 * 5


def test_noPre_without_aggregated_exits(tmp_path, monkeypatch):
    # raw.csv exists but no aggregated_data.csv
    ds = tmp_path / "DS2"; ds.mkdir()  # noqa: E702
    raw = ds / "raw.csv"; raw.write_text("x\n1\n")  # noqa: E702
    monkeypatch.chdir(tmp_path)

    importlib.reload(main_mod)
    sys.argv = ["prog", str(raw), "--noPre"]
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 1


def test_noMod_skips_modeling_but_runs_sims(tmp_path, monkeypatch):
    raw, agg = make_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    importlib.reload(main_mod)
    calls = {'pre': 0, 'bin': 0, 'seq': 0}

    monkeypatch.setattr(main_mod, 'run_preprocessing',
        lambda *args, **kwargs: calls.__setitem__('pre', calls['pre'] + 1))
    monkeypatch.setattr(main_mod, 'run_binary_classification',
        lambda *args, **kwargs: calls.__setitem__('bin', calls['bin'] + 1))
    monkeypatch.setattr(main_mod, 'sequential_simulation',
        lambda **kwargs: calls.__setitem__('seq', calls['seq'] + 1) or [])

    # stub out other sims so they don't error
    monkeypatch.setattr(main_mod, 'continuous_simulation', lambda **k: ([], []))
    monkeypatch.setattr(main_mod, 'parallel_simulation', lambda **k: [])

    sys.argv = ["prog", str(raw), "--noMod"]
    main_mod.main()

    # Preprocessing still runs
    assert calls['pre'] == 1
    # Modeling is skipped
    assert calls['bin'] == 0
    # Simulations should still run at least once
    assert calls['seq'] > 0


def test_noSim_skips_simulations(tmp_path, monkeypatch):
    raw, agg = make_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    importlib.reload(main_mod)
    calls = {'sim': 0}

    # stub out all upstream steps
    monkeypatch.setattr(main_mod, 'run_preprocessing', lambda *a, **k: None)
    monkeypatch.setattr(main_mod, 'run_binary_classification', lambda *a, **k: None)
    monkeypatch.setattr(main_mod, 'run_multiclass_classification', lambda *a, **k: None)
    monkeypatch.setattr(main_mod, 'run_feature_engineering', lambda *a, **k: None)

    # stub sims to increment sim counter
    monkeypatch.setattr(main_mod, 'sequential_simulation',
        lambda **kwargs: calls.__setitem__('sim', calls['sim'] + 1) or [])
    monkeypatch.setattr(main_mod, 'continuous_simulation',
        lambda **kwargs: ([], calls.__setitem__('sim', calls['sim'] + 1) or []))
    monkeypatch.setattr(main_mod, 'parallel_simulation',
        lambda **kwargs: calls.__setitem__('sim', calls['sim'] + 1) or [])

    sys.argv = ["prog", str(raw), "--noSim"]
    main_mod.main()

    # No simulation functions should have been called
    assert calls['sim'] == 0
