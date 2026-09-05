import pytest

from scripts.mc_labeling import AttackMapping, _parse_mapping


def test_parse_mapping_valid():
    mapping = _parse_mapping('192.168.1.192,192.168.1.103,XMasAttack')
    assert mapping == AttackMapping(
        src_ip='192.168.1.192', dst_ip='192.168.1.103', attack_name='XMasAttack'
    )


def test_parse_mapping_strips_whitespace():
    mapping = _parse_mapping(' 192.168.1.192 , 192.168.1.103 , XMasAttack ')
    assert mapping == AttackMapping(
        src_ip='192.168.1.192', dst_ip='192.168.1.103', attack_name='XMasAttack'
    )


def test_parse_mapping_wrong_field_count():
    with pytest.raises(ValueError, match='exactly 3 comma-separated fields'):
        _parse_mapping('192.168.1.192,192.168.1.103')


def test_parse_mapping_invalid_src_ip():
    with pytest.raises(ValueError, match='Invalid source IP'):
        _parse_mapping('not-an-ip,192.168.1.103,XMasAttack')


def test_parse_mapping_invalid_dst_ip():
    with pytest.raises(ValueError, match='Invalid destination IP'):
        _parse_mapping('192.168.1.192,not-an-ip,XMasAttack')


def test_parse_mapping_empty_attack_name():
    with pytest.raises(ValueError, match='attack_name must not be empty'):
        _parse_mapping('192.168.1.192,192.168.1.103, ')


from pathlib import Path

from scripts.mc_labeling import MCLabelConfig, _parse_arguments


def test_parse_arguments_single_mapping():
    config = _parse_arguments([
        'data.csv',
        '--mapping', '192.168.1.192,192.168.1.103,XMasAttack',
    ])
    assert config == MCLabelConfig(
        dataset_path=Path('data.csv'),
        mappings=(AttackMapping('192.168.1.192', '192.168.1.103', 'XMasAttack'),),
    )


def test_parse_arguments_multiple_mappings():
    config = _parse_arguments([
        'data.csv',
        '--mapping', '192.168.1.192,192.168.1.103,XMasAttack',
        '--mapping', '10.0.0.1,10.0.0.2,PortScan',
    ])
    assert config.mappings == (
        AttackMapping('192.168.1.192', '192.168.1.103', 'XMasAttack'),
        AttackMapping('10.0.0.1', '10.0.0.2', 'PortScan'),
    )


def test_parse_arguments_requires_at_least_one_mapping():
    with pytest.raises(SystemExit):
        _parse_arguments(['data.csv'])
