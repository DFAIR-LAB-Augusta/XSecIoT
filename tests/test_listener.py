from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def listener_module(monkeypatch: pytest.MonkeyPatch):
    """
    Import the listener module in a way that's friendly to Flask testing.

    Adjust the import below to your actual module path, e.g.:
      - from src.listener import app, run_server
      - from src.utils.listener import app, run_server
    """
    import src.core.listener as listener

    listener.data_callback = None
    return listener


@pytest.fixture
def client(listener_module):
    listener_module.app.config.update(TESTING=True)
    return listener_module.app.test_client()


def test_receive_csv_ok_no_callback(client, listener_module):
    csv_payload = 'a,b\n1,2\n3,4\n'

    resp = client.post('/', data=csv_payload, content_type='text/plain')
    assert resp.status_code == 200
    assert resp.data.decode('utf-8') == 'CSV received'


def test_receive_csv_ok_invokes_callback_with_dataframe(client, listener_module):
    seen: dict[str, Any] = {}

    def cb(df):
        seen['df'] = df

    listener_module.data_callback = cb

    csv_payload = 'x,y\n10,20\n'
    resp = client.post('/', data=csv_payload, content_type='text/plain')

    assert resp.status_code == 200
    assert 'df' in seen

    df = seen['df']
    assert list(df.columns) == ['x', 'y']
    assert df.shape == (1, 2)
    assert int(df.iloc[0]['x']) == 10
    assert int(df.iloc[0]['y']) == 20


def test_receive_csv_bad_payload_returns_400(client, listener_module, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise ValueError('nope')

    monkeypatch.setattr(listener_module.pd, 'read_csv', boom)

    resp = client.post('/', data='a,b\n1,2\n', content_type='text/plain')
    assert resp.status_code == 400
    body = resp.data.decode('utf-8')
    assert body.startswith('Error:')
    assert 'nope' in body


def test_run_server_sets_callback_and_calls_app_run(monkeypatch: pytest.MonkeyPatch, listener_module):
    called: dict[str, Any] = {}

    def fake_run(*, host: str, port: int, **kwargs: Any):
        called['host'] = host
        called['port'] = port
        called['kwargs'] = kwargs

    monkeypatch.setattr(listener_module.app, 'run', fake_run)

    def cb(df):
        return None

    listener_module.run_server(callback=cb, host='0.0.0.0', port=9999)

    assert listener_module.data_callback is cb
    assert called['host'] == '0.0.0.0'
    assert called['port'] == 9999
