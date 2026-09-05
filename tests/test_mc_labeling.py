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


from scripts.mc_labeling import _validate_inputs


def test_validate_inputs_missing_file(tmp_path):
    missing = tmp_path / 'nope.csv'
    with pytest.raises(FileNotFoundError):
        _validate_inputs(missing, (AttackMapping('1.2.3.4', '5.6.7.8', 'X'),))


def test_validate_inputs_non_csv(tmp_path):
    bad = tmp_path / 'data.txt'
    bad.write_text('not a csv')
    with pytest.raises(ValueError, match='Only CSV files are supported'):
        _validate_inputs(bad, (AttackMapping('1.2.3.4', '5.6.7.8', 'X'),))


def test_validate_inputs_duplicate_pair_conflicting_names(tmp_path):
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('src_ip,dst_ip\n1.2.3.4,5.6.7.8\n')
    mappings = (
        AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack'),
        AttackMapping('1.2.3.4', '5.6.7.8', 'PortScan'),
    )
    with pytest.raises(ValueError, match='duplicate mapping'):
        _validate_inputs(csv_path, mappings)


def test_validate_inputs_accepts_valid_config(tmp_path):
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('src_ip,dst_ip\n1.2.3.4,5.6.7.8\n')
    _validate_inputs(csv_path, (AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack'),))


import pandas as pd

from scripts.mc_labeling import _add_multiclass_labels


def test_add_multiclass_labels_assigns_attack_and_benign(tmp_path):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({
        'src_ip': ['1.2.3.4', '1.2.3.4', '9.9.9.9'],
        'dst_ip': ['5.6.7.8', '5.6.7.8', '9.9.9.8'],
        'flow_duration': [1, 2, 3],
    }).to_csv(csv_path, index=False)

    mapping = AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack')
    output_path = _add_multiclass_labels(csv_path, (mapping,))

    assert output_path == tmp_path / 'flows_mc_labeled.csv'
    result = pd.read_csv(output_path)
    assert result['MC_Label'].tolist() == ['XMasAttack', 'XMasAttack', 'Benign']


def test_add_multiclass_labels_multiple_mappings(tmp_path):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({
        'src_ip': ['1.2.3.4', '10.0.0.1', '9.9.9.9'],
        'dst_ip': ['5.6.7.8', '10.0.0.2', '9.9.9.8'],
    }).to_csv(csv_path, index=False)

    mappings = (
        AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack'),
        AttackMapping('10.0.0.1', '10.0.0.2', 'PortScan'),
    )
    output_path = _add_multiclass_labels(csv_path, mappings)
    result = pd.read_csv(output_path)
    assert result['MC_Label'].tolist() == ['XMasAttack', 'PortScan', 'Benign']


def test_add_multiclass_labels_missing_ip_columns(tmp_path):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="must contain 'src_ip' and 'dst_ip'"):
        _add_multiclass_labels(csv_path, (AttackMapping('1.2.3.4', '5.6.7.8', 'X'),))


from scripts.mc_labeling import main


def test_main_end_to_end_success(tmp_path, capsys):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({
        'src_ip': ['1.2.3.4', '9.9.9.9'],
        'dst_ip': ['5.6.7.8', '9.9.9.8'],
    }).to_csv(csv_path, index=False)

    main([str(csv_path), '--mapping', '1.2.3.4,5.6.7.8,XMasAttack'])

    captured = capsys.readouterr()
    expected_output = tmp_path / 'flows_mc_labeled.csv'
    assert str(expected_output) in captured.out
    assert expected_output.exists()


def test_main_exits_nonzero_on_missing_file(tmp_path, capsys):
    missing = tmp_path / 'nope.csv'
    with pytest.raises(SystemExit) as exc_info:
        main([str(missing), '--mapping', '1.2.3.4,5.6.7.8,XMasAttack'])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert 'Error:' in captured.err


def test_main_exits_nonzero_on_invalid_mapping(tmp_path, capsys):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({'src_ip': ['1.2.3.4'], 'dst_ip': ['5.6.7.8']}).to_csv(csv_path, index=False)

    with pytest.raises(SystemExit) as exc_info:
        main([str(csv_path), '--mapping', 'not-an-ip,5.6.7.8,XMasAttack'])

    assert exc_info.value.code == 1
