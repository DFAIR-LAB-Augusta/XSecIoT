import subprocess
import sys

from firce.cli import _build_config_from_args
from firce.utils.arg_parser import parse_sim_args


def test_cwd_independent_import(tmp_path):
    result = subprocess.run([sys.executable, '-c', 'import firce.cli'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_novelty_cli_flags_land_on_config(monkeypatch, tmp_path):
    data = tmp_path / 'data.csv'
    data.write_text('a\n1\n')
    argv = [
        'prog',
        str(data),
        str(data),
        '--novelty',
        '--noveltyTau',
        '0.7',
        '--noveltyAlpha',
        '0.2',
        '--noveltySelectiveMode',
        'windowed',
        '--noveltyExplainMethod',
        'lime',
        '--noveltyLlmBackendType',
        'ollama',
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    config = _build_config_from_args(parse_sim_args())
    assert config.novelty_enabled is True
    assert config.novelty_tau == 0.7
    assert config.novelty_alpha == 0.2
    assert config.novelty_selective_mode == 'windowed'
    assert config.novelty_explain_method == 'lime'
    assert config.novelty_llm_backend_type == 'ollama'
