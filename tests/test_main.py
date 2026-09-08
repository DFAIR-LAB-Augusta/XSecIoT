import sys

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
