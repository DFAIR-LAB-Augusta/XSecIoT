import pandas as pd
import pytest

from scripts import capex_labeling

GT_HEADER = 'session_id,device,target_ip,label,attempt,start_unix,end_unix\n'
# 2024-01-01 00:00:00 UTC
BASE = 1704067200.0


def _write_gt(path, rows):
    lines = [f'{i},dev,{ip},{label},1,{BASE + s},{BASE + e}' for i, (ip, label, s, e) in enumerate(rows)]
    path.write_text(GT_HEADER + '\n'.join(lines) + '\n')
    return path


def _write_flows(path, rows):
    lines = ['src_ip,dst_ip,timestamp,flow_duration']
    for src, dst, offset, dur in rows:
        ts = pd.Timestamp(BASE + offset, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f'{src},{dst},{ts},{dur}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def _run(tmp_path, flows, gt, extra=()):
    f = _write_flows(tmp_path / 'flows.csv', flows)
    g = _write_gt(tmp_path / 'gt.csv', gt)
    capex_labeling.main([str(f), str(g), '--flowTz', 'UTC', *extra])
    return pd.read_csv(tmp_path / 'flows_mc_labeled.csv')


def test_back_to_back_attacks_one_device(tmp_path):
    gt = [('10.0.0.5', 'UDP_Flood', 0, 10), ('10.0.0.5', 'TCP_SYN_Flood', 20, 30)]
    flows = [('1.1.1.1', '10.0.0.5', 2, 3), ('10.0.0.5', '1.1.1.1', 22, 3)]
    df = _run(tmp_path, flows, gt)
    assert list(df['MC_Label']) == ['UDP_Flood', 'TCP_SYN_Flood']
    assert not df['MC_Partial'].any()


def test_partial_overlap_picks_larger_overlap(tmp_path):
    gt = [('10.0.0.5', 'UDP_Flood', 0, 10), ('10.0.0.5', 'TCP_SYN_Flood', 10, 30)]
    flows = [('1.1.1.1', '10.0.0.5', 8, 10), ('1.1.1.1', '10.0.0.5', 9, 3)]
    df = _run(tmp_path, flows, gt)
    assert list(df['MC_Label']) == ['TCP_SYN_Flood', 'TCP_SYN_Flood']
    assert df['MC_Partial'].all()


def test_tie_takes_earlier_window(tmp_path):
    gt = [('10.0.0.5', 'TCP_SYN_Flood', 10, 20), ('10.0.0.5', 'UDP_Flood', 0, 10)]
    df = _run(tmp_path, [('1.1.1.1', '10.0.0.5', 5, 10)], gt)
    assert df['MC_Label'][0] == 'UDP_Flood'
    assert bool(df['MC_Partial'][0])


def test_partial_flow_touching_one_window_edge(tmp_path):
    gt = [('10.0.0.5', 'UDP_Flood', 0, 10)]
    df = _run(tmp_path, [('1.1.1.1', '10.0.0.5', 8, 5)], gt)
    assert df['MC_Label'][0] == 'UDP_Flood'
    assert bool(df['MC_Partial'][0])


def test_benign_gaps(tmp_path):
    gt = [('10.0.0.5', 'UDP_Flood', 10, 20)]
    flows = [('1.1.1.1', '10.0.0.5', 0, 2), ('1.1.1.1', '10.0.0.5', 40, 2)]
    df = _run(tmp_path, flows, gt)
    assert list(df['MC_Label']) == ['Benign', 'Benign']
    assert not df['MC_Partial'].any()


def test_non_target_ip_in_window_is_benign(tmp_path):
    gt = [('10.0.0.5', 'UDP_Flood', 0, 10)]
    df = _run(tmp_path, [('1.1.1.1', '10.0.0.9', 2, 3)], gt)
    assert df['MC_Label'][0] == 'Benign'


def test_non_canonical_label_rejected(tmp_path, capsys):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', 'NotAnAttack', 0, 10)])
    with pytest.raises(SystemExit) as exc:
        capex_labeling.main([str(f), str(g), '--flowTz', 'UTC'])
    assert exc.value.code == 1
    assert 'NotAnAttack' in capsys.readouterr().err


@pytest.mark.parametrize('label', ['Config_Tamper_Setup_CGI', 'Config_Tamper_TR064_SOAP'])
def test_config_tamper_label_excluded(tmp_path, capsys, label):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', label, 0, 10)])
    with pytest.raises(SystemExit) as exc:
        capex_labeling.main([str(f), str(g), '--flowTz', 'UTC'])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert label in err
    assert 'excluded from Direction 3' in err
    assert '#194' in err


def test_missing_gt_column(tmp_path, capsys):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = tmp_path / 'gt.csv'
    g.write_text('session_id,device,target_ip,label,attempt,start_unix\n1,d,10.0.0.5,UDP_Flood,1,5\n')
    with pytest.raises(SystemExit):
        capex_labeling.main([str(f), str(g), '--flowTz', 'UTC'])
    assert 'end_unix' in capsys.readouterr().err


def test_end_before_start_rejected(tmp_path, capsys):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', 'UDP_Flood', 10, 5)])
    with pytest.raises(SystemExit):
        capex_labeling.main([str(f), str(g), '--flowTz', 'UTC'])
    assert 'end_unix' in capsys.readouterr().err


@pytest.mark.parametrize('column', ['target_ip', 'start_unix', 'end_unix'])
def test_blank_gt_field_rejected(tmp_path, capsys, column):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', 'UDP_Flood', 0, 10)])
    frame = pd.read_csv(g, dtype=str)
    frame[column] = ''
    frame.to_csv(g, index=False)
    with pytest.raises(SystemExit) as excinfo:
        capex_labeling.main([str(f), str(g), '--flowTz', 'UTC'])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert column in err
    assert '1 row(s)' in err


def test_blank_gt_whitespace_padded_target_ip_still_labels(tmp_path):
    out = _run(tmp_path, [('1.1.1.1', '10.0.0.5', 2, 1)], [(' 10.0.0.5 ', 'UDP_Flood', 0, 10)])
    assert out['MC_Label'].tolist() == ['UDP_Flood']


def test_unknown_timezone(tmp_path, capsys):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', 'UDP_Flood', 0, 10)])
    with pytest.raises(SystemExit):
        capex_labeling.main([str(f), str(g), '--flowTz', 'Mars/Olympus'])
    assert 'Mars/Olympus' in capsys.readouterr().err


def test_flow_tz_conversion(tmp_path):
    # Window is 00:00:00-00:00:10 UTC. Naive stamps written as UTC wall clock read 19:00 the prior day in New York.
    gt = [('10.0.0.5', 'UDP_Flood', 0, 10)]
    f = tmp_path / 'flows.csv'
    f.write_text(
        'src_ip,dst_ip,timestamp,flow_duration\n'
        '1.1.1.1,10.0.0.5,2023-12-31 19:00:02,1\n'
        '1.1.1.1,10.0.0.5,2024-01-01 00:00:02,1\n'
    )
    g = _write_gt(tmp_path / 'gt.csv', gt)
    capex_labeling.main([str(f), str(g), '--flowTz', 'America/New_York'])
    df = pd.read_csv(tmp_path / 'flows_mc_labeled.csv')
    assert list(df['MC_Label']) == ['UDP_Flood', 'Benign']


def test_default_flow_tz_host_zone_honours_dst(tmp_path, monkeypatch):
    # 2026-07-04 12:00:00 New York (EDT) is 16:00:00 UTC (1783180800), not 17:00:00 as a fixed EST offset gives.
    monkeypatch.setenv('TZ', 'America/New_York')
    f = tmp_path / 'flows.csv'
    f.write_text('src_ip,dst_ip,timestamp,flow_duration\n1.1.1.1,10.0.0.5,2026-07-04 12:00:00,1\n')

    def label(start):
        g = tmp_path / 'gt.csv'
        g.write_text(GT_HEADER + f'0,dev,10.0.0.5,UDP_Flood,1,{start},{start + 10}\n')
        capex_labeling.main([str(f), str(g)])
        return pd.read_csv(tmp_path / 'flows_mc_labeled.csv')['MC_Label'][0]

    assert label(1783180800) == 'UDP_Flood'
    assert label(1783184400) == 'Benign'


def test_dst_transition_flows_dropped_and_reported(tmp_path, capsys):
    # 2026-11-01 01:30 is ambiguous (fall-back) and 2026-03-08 02:30 is nonexistent (spring-forward) in New York.
    f = tmp_path / 'flows.csv'
    f.write_text(
        'src_ip,dst_ip,timestamp,flow_duration\n'
        '1.1.1.1,10.0.0.5,2024-01-01 00:00:02,1\n'
        '1.1.1.1,10.0.0.5,2026-11-01 01:30:00,1\n'
        '1.1.1.1,10.0.0.5,2026-03-08 02:30:00,1\n'
        '1.1.1.1,10.0.0.7,2024-01-01 00:00:02,1\n'
    )
    # 2024-01-01 00:00:00 New York is 05:00:00 UTC.
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', 'UDP_Flood', 5 * 3600, 5 * 3600 + 10)])
    capex_labeling.main([str(f), str(g), '--flowTz', 'America/New_York'])
    captured = capsys.readouterr()
    df = pd.read_csv(tmp_path / 'flows_mc_labeled.csv')
    assert list(df['timestamp']) == ['2024-01-01 00:00:02', '2024-01-01 00:00:02']
    assert list(df['MC_Label']) == ['UDP_Flood', 'Benign']
    assert 'Warning' in captured.err
    assert 'Dropped (DST transition): 2' in captured.out


def test_dst_transition_none_reports_zero(tmp_path, capsys):
    _run(tmp_path, [('1.1.1.1', '10.0.0.5', 2, 1)], [('10.0.0.5', 'UDP_Flood', 0, 10)])
    captured = capsys.readouterr()
    assert 'Dropped (DST transition): 0' in captured.out
    assert 'Warning' not in captured.err


def test_report_counts_and_timezone(tmp_path, capsys):
    gt = [('10.0.0.5', 'UDP_Flood', 0, 10), ('10.0.0.5', 'Slowloris', 20, 30)]
    _run(tmp_path, [('1.1.1.1', '10.0.0.5', 2, 1), ('1.1.1.1', '10.0.0.7', 2, 1)], gt)
    out = capsys.readouterr().out
    assert 'UTC' in out
    assert 'Benign: 1' in out
    assert 'UDP_Flood: 1' in out
    assert 'Slowloris: 0' in out
    assert 'MC_Partial: 0' in out


def test_columns_preserved_and_custom_output(tmp_path):
    f = _write_flows(tmp_path / 'flows.csv', [('1.1.1.1', '10.0.0.5', 0, 1)])
    g = _write_gt(tmp_path / 'gt.csv', [('10.0.0.5', 'UDP_Flood', 0, 10)])
    out = tmp_path / 'custom.csv'
    capex_labeling.main([str(f), str(g), '--flowTz', 'UTC', '--output', str(out)])
    df = pd.read_csv(out)
    assert list(df.columns) == ['src_ip', 'dst_ip', 'timestamp', 'flow_duration', 'MC_Label', 'MC_Partial']
