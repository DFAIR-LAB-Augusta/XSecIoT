import os

from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


def test_summarize_timings_empty(caplog) -> None:
    with caplog.at_level('INFO'):
        sim._summarize_timings('Timing', [])
    assert 'Timing: No timings recorded.' in caplog.text


def test_log_results_saves_accuracy_plot_fixed_chunking(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    (tmp_path / 'logging' / 'chunk_size_16').mkdir(parents=True, exist_ok=True)

    saved = {'paths': []}

    def fake_savefig(path, *args, **kwargs):
        saved['paths'].append(str(path))

    monkeypatch.setattr(sim.plt, 'savefig', fake_savefig)
    monkeypatch.setattr(sim.time, 'perf_counter', lambda: 123.0)

    config = SimpleNamespace(
        use_adaptive_chunking=False,
        chunk_size=16,
        model_variant=sim.ModelVariant.KNN,
        ce_type=sim.CEType.CCE,
        model_type=sim.ModelType.BINARY,
    )

    perf = sim.PerformanceStats()
    perf.correct_log = [True] * 120 + [False] * 30
    perf.drift_detected_indices = []
    perf.drift_intervals = []
    perf.iteration_times = [0.01, 0.02]
    perf.drift_times = [0.001]

    sim._log_results(config, overall=0.0, perf_stats=perf)  # type: ignore

    assert saved['paths'], 'expected at least one plot to be saved'
    assert any('accuracy_plot.png' in p for p in saved['paths'])
