import sys

from fire.models import _parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv'])
    args = _parse_args()

    assert args.aggregated_file == 'data.csv'
    assert not args.unsw
    assert not args.pca
    assert not args.shap
    assert not args.lime


def test_parse_args_xai_flags(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv', '--shap', '--lime', '--pca'])
    args = _parse_args()

    assert args.shap
    assert args.lime
    assert args.pca
