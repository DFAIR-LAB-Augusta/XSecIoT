import os

from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


def test_parse_args_minimal(monkeypatch, tmp_path: Path) -> None:
    agg = tmp_path / 'agg.csv'
    flows = tmp_path / 'flows.csv'
    monkeypatch.setattr(
        'sys.argv',
        [
            'ce_simulation.py',
            str(agg),
            str(flows),
        ],
    )

    args = sim._parse_args()

    assert args.aggregated_file == agg
    assert args.flows_file == flows
    assert args.chunk_size == 1000
    assert args.max_rows == 10000
    assert args.use_pca is False
    assert args.log2File is False
    assert args.debug is False


def test_configure_logging_creates_run_log_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = SimpleNamespace(
        debug=True,
        log_to_file=True,
        use_adaptive_chunking=False,
        chunk_size=123,
        model_variant=sim.ModelVariant.KNN,
        ce_type=sim.CEType.CCE,
        model_type=sim.ModelType.BINARY,
    )

    sim._configure_logging(config)  # type: ignore

    expected = (
        tmp_path
        / 'logging'
        / 'chunk_size_123'
        / f'{config.model_variant.value}_{config.ce_type.value}_{config.model_type.value}_run.log'
    )
    assert expected.exists()


def test_ensure_models_exist_calls_binary_trainer(monkeypatch, tmp_path: Path) -> None:
    called = {'bin': 0, 'mc': 0}

    def fake_train_bin(*args, **kwargs):
        called['bin'] += 1

    def fake_train_mc(*args, **kwargs):
        called['mc'] += 1

    monkeypatch.setattr(sim, 'train_ce_binary', fake_train_bin)
    monkeypatch.setattr(sim, 'train_ce_multiclass', fake_train_mc)

    config = SimpleNamespace(
        aggregated_path=tmp_path / 'ds' / 'agg.csv',
        model_type=sim.ModelType.BINARY,
        model_variant=sim.ModelVariant.KNN,
    )
    config.aggregated_path.parent.mkdir(parents=True, exist_ok=True)

    sim._ensure_models_exist(config, sim.PerformanceStats())  # type: ignore
    assert called['bin'] == 1
    assert called['mc'] == 0
