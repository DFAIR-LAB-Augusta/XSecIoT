import numpy as np
import pytest

from firce.novelty.evaluation import SESSION_TYPES, evaluate_detection_delay


def test_detection_delay_after_change_point():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    flags = np.array([False, True, False, False, True, True])

    result = evaluate_detection_delay(timestamps, flags, 'attack', change_point=2.0)

    assert result['session_type'] == 'attack'
    assert result['detected'] is True
    assert result['delay_seconds'] == pytest.approx(2.0)
    assert result['delay_flows'] == 2
    assert result['false_alerts'] == 1
    assert result['false_alert_rate'] == pytest.approx(0.5)
    assert result['n_flows'] == 6


def test_detection_delay_flag_on_first_post_change_flow_is_zero_flows():
    result = evaluate_detection_delay(np.array([0.0, 1.0, 2.0]), np.array([False, True, True]), 'attack', 1.0)

    assert result['delay_flows'] == 0
    assert result['delay_seconds'] == pytest.approx(0.0)


def test_detection_delay_sorts_timestamps_and_keeps_flags_aligned():
    timestamps = np.array([5.0, 0.0, 3.0, 1.0])
    flags = np.array([True, False, False, True])

    result = evaluate_detection_delay(timestamps, flags, 'attack', change_point=2.0)

    assert result['false_alerts'] == 1
    assert result['delay_seconds'] == pytest.approx(3.0)
    assert result['delay_flows'] == 1


def test_detection_delay_flags_only_before_change_point_are_false_alerts():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    flags = np.array([True, True, False, False])

    result = evaluate_detection_delay(timestamps, flags, 'attack', change_point=2.0)

    assert result['detected'] is False
    assert result['delay_seconds'] is None
    assert result['delay_flows'] is None
    assert result['false_alerts'] == 2
    assert result['false_alert_rate'] == pytest.approx(1.0)


def test_detection_delay_never_flagged():
    result = evaluate_detection_delay(np.array([0.0, 1.0, 2.0]), np.zeros(3, dtype=bool), 'attack', 1.0)

    assert result['detected'] is False
    assert result['delay_seconds'] is None
    assert result['false_alerts'] == 0
    assert result['false_alert_rate'] == 0.0


def test_detection_delay_attack_change_point_before_all_flows_has_empty_window():
    result = evaluate_detection_delay(np.array([1.0, 2.0]), np.array([False, True]), 'attack', 0.0)

    assert result['detected'] is True
    assert result['delay_flows'] == 1
    assert result['false_alerts'] == 0
    assert result['false_alert_rate'] == 0.0


@pytest.mark.parametrize(['session_type'], [['stable'], ['benign']])
def test_detection_delay_non_attack_sessions_count_every_flag_as_false_alert(session_type):
    result = evaluate_detection_delay(
        np.array([2.0, 0.0, 1.0, 3.0]), np.array([True, False, True, False]), session_type
    )

    assert result['session_type'] == session_type
    assert result['detected'] is None
    assert result['delay_seconds'] is None
    assert result['delay_flows'] is None
    assert result['false_alerts'] == 2
    assert result['false_alert_rate'] == pytest.approx(0.5)
    assert result['n_flows'] == 4


def test_detection_delay_session_types_constant():
    assert SESSION_TYPES == ('stable', 'attack', 'benign')


@pytest.mark.parametrize(
    ['session_type', 'change_point'],
    [['attack', None], ['stable', 1.0], ['benign', 1.0], ['unknown', None]],
)
def test_detection_delay_rejects_invalid_session_arguments(session_type, change_point):
    with pytest.raises(ValueError):
        evaluate_detection_delay(np.array([0.0, 1.0]), np.array([False, True]), session_type, change_point)


def test_detection_delay_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_detection_delay(np.array([0.0, 1.0]), np.array([False]), 'stable')
