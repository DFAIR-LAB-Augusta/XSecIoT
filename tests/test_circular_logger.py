from __future__ import annotations

import pytest

from src.core.circular_logger import CircularDequeLogger


def test_append_without_columns_and_to_dataframe() -> None:
    log = CircularDequeLogger(max_rows=5, columns=None)
    log.append([1, 'a'])
    log.append([2, 'b'])

    df = log.to_dataframe()
    assert df.shape == (2, 2)
    assert df.iloc[0].tolist() == [1, 'a']
    assert df.iloc[1].tolist() == [2, 'b']


def test_append_with_columns_sets_schema_and_to_dataframe_columns() -> None:
    cols = ['x', 'y', 'z']
    log = CircularDequeLogger(max_rows=10, columns=cols)

    log.append([1, 2, 3])
    log.append([4, 5, 6])

    df = log.to_dataframe()
    assert list(df.columns) == cols
    assert df.shape == (2, 3)
    assert df.iloc[0].to_dict() == {'x': 1, 'y': 2, 'z': 3}
    assert df.iloc[1].to_dict() == {'x': 4, 'y': 5, 'z': 6}


def test_append_schema_mismatch_raises_value_error() -> None:
    log = CircularDequeLogger(max_rows=10, columns=['a', 'b', 'c'])
    with pytest.raises(ValueError) as exc:
        log.append([1, 2])
    assert 'Row width' in str(exc.value)
    assert 'schema width' in str(exc.value)


def test_deque_eviction_respects_max_rows() -> None:
    log = CircularDequeLogger(max_rows=3, columns=['n'])

    log.append([1])
    log.append([2])
    log.append([3])
    log.append([4])

    df = log.to_dataframe()
    assert df.shape == (3, 1)
    assert df['n'].tolist() == [2, 3, 4]


def test_flush_is_noop_and_close_calls_flush(monkeypatch) -> None:
    log = CircularDequeLogger(max_rows=3, columns=['n'])

    called = {'flush': 0}

    def _flush() -> None:
        called['flush'] += 1

    monkeypatch.setattr(log, 'flush', _flush)

    log.close()
    assert called['flush'] == 1


def test_context_manager_calls_close(monkeypatch) -> None:
    called = {'close': 0}

    def _close(self) -> None:
        called['close'] += 1

    monkeypatch.setattr(CircularDequeLogger, 'close', _close, raising=True)

    with CircularDequeLogger(max_rows=2, columns=['a']) as log:
        log.append([1])

    assert called['close'] == 1


@pytest.mark.slow
def test_largeish_buffer_roundtrip_not_too_big() -> None:
    n = 5000
    log = CircularDequeLogger(max_rows=n, columns=['i', 'v'])
    for i in range(n):
        log.append([i, i * 2])

    df = log.to_dataframe()
    assert df.shape == (n, 2)
    assert df.iloc[0].tolist() == [0, 0]
    assert df.iloc[-1].tolist() == [n - 1, (n - 1) * 2]
