import pytest

from pydantic import ValidationError

from firce.drift_monitor.cade_config import CadeMonitorConfig


def test_cade_monitor_config_accepts_valid_dims():
    config = CadeMonitorConfig(dims=[10, 5, 2])

    assert config.dims == [10, 5, 2]
    assert config.batch_size == 64
    assert config.mad_threshold == 3.5


def test_cade_monitor_config_rejects_single_element_dims():
    # pydantic's own Field(min_length=2) constraint fires before the custom
    # field_validator, so its built-in "too_short" message wins here.
    with pytest.raises(ValidationError, match='at least 2 items'):
        CadeMonitorConfig(dims=[10])


def test_cade_monitor_config_rejects_non_positive_dims():
    with pytest.raises(ValidationError, match='must be positive'):
        CadeMonitorConfig(dims=[10, 0])


def test_cade_monitor_config_rejects_batch_size_not_multiple_of_four():
    with pytest.raises(ValidationError, match='multiple of 4'):
        CadeMonitorConfig(dims=[10, 2], batch_size=5)


def test_cade_monitor_config_rejects_out_of_range_ratio():
    with pytest.raises(ValidationError, match=r'must be in \[0, 1\]'):
        CadeMonitorConfig(dims=[10, 2], min_drift_ratio=1.5)


def test_cade_monitor_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        CadeMonitorConfig(dims=[10, 2], not_a_real_field=1)


def test_cade_monitor_config_is_frozen():
    config = CadeMonitorConfig(dims=[10, 2])

    with pytest.raises(ValidationError):
        config.batch_size = 128
