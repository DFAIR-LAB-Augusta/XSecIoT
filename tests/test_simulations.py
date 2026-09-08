import sys

import numpy as np
import pandas as pd

from fire.simulations import _get_dataset_name, _parse_args, preprocess_chunk


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'agg.csv'])
    args = _parse_args()

    assert args.aggregated_file == 'agg.csv'
    assert args.mode == 'sequential'
    assert args.model_type == 'binary'
    assert args.model_variant == 'dt'
    assert args.chunk_size == 1000
    assert not args.unsw


def test_parse_args_overrides(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv', ['prog', 'agg.csv', '--mode', 'parallel', '--model_type', 'multi', '--model_variant', 'rf', '--unsw']
    )
    args = _parse_args()

    assert args.mode == 'parallel'
    assert args.model_type == 'multi'
    assert args.model_variant == 'rf'
    assert args.unsw


def test_get_dataset_name():
    assert _get_dataset_name('/foo/BAR/agg.csv') == 'BAR'


def test_preprocess_chunk_drops_columns_and_fills_na():
    df = pd.DataFrame({'a': [1, np.nan], 'b': [np.nan, 2], 'drop': [9, 9]})

    cleaned = preprocess_chunk(df, ['drop'])

    assert 'drop' not in cleaned.columns
    assert not cleaned.isna().any().any()
