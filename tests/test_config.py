from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from pydantic import ValidationError

from src.core.config import (
    AdaptiveChunkConfig,
    CEType,
    ModelType,
    ModelVariant,
    SimulationConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


def _touch(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text('x', encoding='utf-8')
    return p


def _base_payload(tmp_path: Path) -> dict[str, Any]:
    agg = _touch(tmp_path, 'agg.csv')
    flows = _touch(tmp_path, 'flows.csv')
    return {
        'model_type': ModelType.BINARY,
        'model_variant': ModelVariant.RF,
        'ce_type': CEType.NONE,
        'aggregated_path': agg,
        'flows_path': flows,
        'device': torch.device('cpu'),
    }


def test_simulation_config_happy_path_and_device_serialization(tmp_path: Path) -> None:
    cfg = SimulationConfig(**_base_payload(tmp_path))
    dumped = cfg.model_dump()
    assert dumped['device'] == 'cpu'
    assert cfg.model_type is ModelType.BINARY
    assert cfg.model_variant is ModelVariant.RF
    assert cfg.ce_type is CEType.NONE


def test_simulation_config_enum_coercion_from_strings(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload.update({
        'model_type': 'multi',
        'model_variant': 'rf',
        'ce_type': 'approx_tce',
    })

    cfg = SimulationConfig.model_validate(payload)
    assert cfg.model_type is ModelType.MULTI
    assert cfg.model_variant is ModelVariant.RF
    assert cfg.ce_type is CEType.APPROX_TCE


def test_simulation_config_forbids_extra_fields(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload['totally_not_a_field'] = 123

    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)


def test_simulation_config_threshold_validation(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)

    payload['threshold'] = -0.01
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)

    payload['threshold'] = 1.01
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)


def test_simulation_config_positive_int_validation(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)

    payload['chunk_size'] = 0
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)

    payload = _base_payload(tmp_path)
    payload['max_rows'] = 0
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)


def test_simulation_config_path_must_exist_and_be_file(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)

    payload['aggregated_path'] = tmp_path / 'does_not_exist.csv'
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)

    d = tmp_path / 'a_dir'
    d.mkdir()
    payload = _base_payload(tmp_path)
    payload['flows_path'] = d
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)


def test_simulation_config_ce_kwargs_validation(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)

    payload['ce_kwargs'] = {'folds': 1}
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)

    payload = _base_payload(tmp_path)
    payload['ce_kwargs'] = {'significance': 1.0}
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)

    payload = _base_payload(tmp_path)
    payload['ce_kwargs'] = {'significance': 0.0}
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(payload)


def test_adaptive_chunk_config_validators() -> None:
    with pytest.raises(ValidationError):
        AdaptiveChunkConfig(init_chunk_size=-1)

    with pytest.raises(ValidationError):
        AdaptiveChunkConfig(ema_decay=1.1)

    with pytest.raises(ValidationError):
        AdaptiveChunkConfig(min_chunk_size=10, max_chunk_size=5)

    with pytest.raises(ValidationError):
        AdaptiveChunkConfig(min_chunk_size=10, max_chunk_size=20, init_chunk_size=25)
