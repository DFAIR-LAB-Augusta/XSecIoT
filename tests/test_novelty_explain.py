import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.novelty.explain import select_events_to_explain


def test_select_events_unknown_only_returns_exactly_the_flagged_indices():
    flags = np.array([False, True, False, True, True])

    selected = select_events_to_explain(flags, mode='unknown_only')

    assert selected.tolist() == [1, 3, 4]


def test_select_events_sampled_respects_sample_rate_and_is_reproducible_with_rng():
    flags = np.zeros(100, dtype=bool)
    rng = np.random.default_rng(0)

    selected = select_events_to_explain(flags, mode='sampled', sample_rate=0.2, rng=rng)

    assert 15 <= len(selected) <= 25  # ~20% of 100, allow sampling variance
    assert len(set(selected.tolist())) == len(selected)  # no duplicates
    assert np.all((selected >= 0) & (selected < 100))


def test_select_events_windowed_picks_one_index_per_window_preferring_flagged():
    flags = np.array([False, False, True, False, False, False, False, False, False, False])

    selected = select_events_to_explain(flags, mode='windowed', window_size=5)

    # Window 0 (indices 0-4): index 2 is flagged -> picked.
    # Window 1 (indices 5-9): none flagged -> first index of window (5) picked.
    assert selected.tolist() == [2, 5]


def test_select_events_unknown_mode_raises_value_error():
    flags = np.array([True, False])

    with pytest.raises(ValueError, match='mode'):
        select_events_to_explain(flags, mode='not_a_real_mode')
