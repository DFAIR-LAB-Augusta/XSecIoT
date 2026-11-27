import json

import numpy as np
import pytest

from firce.novelty import llm_reporting
from firce.novelty.llm_reporting import TransformersLocalBackend, create_local_llm_backend
from firce.novelty.template_report import (
    FEATURE_PHRASES,
    TemplateReportBackend,
    generate_template_report,
)
from firce.runtime.constants import FINAL_LOG_COLUMNS
from firce.runtime.inference import _generate_novelty_reports, _score_chunk_novelty
from tests.test_inference import _make_novelty_test_runtime

_XAI = {
    'predicted_class': 'Benign',
    'contributions': {'flow_duration': 1.2, 'syn_flag_cnt': -0.8, 'f3': 0.5, 'tot_bwd_pkts': -0.1},
}
_CONTEXT = {'max_softmax': 0.55, 'tau': 0.6, 'alpha': 0.3}


def test_template_report_feature_phrases_cover_final_log_columns():
    assert len(FEATURE_PHRASES) >= 20
    assert set(FEATURE_PHRASES) <= set(FINAL_LOG_COLUMNS)
    assert FEATURE_PHRASES['flow_duration'] == ('long flow duration', 'short flow duration')
    assert FEATURE_PHRASES['syn_flag_cnt'] == ('many SYN flags', 'few SYN flags')


def test_template_report_is_byte_identical_across_calls():
    first = generate_template_report(_XAI, _CONTEXT)
    second = generate_template_report(_XAI, _CONTEXT)

    assert json.dumps(first) == json.dumps(second)
    assert set(first) == {'summary', 'suggested_label', 'raw_output'}
    assert first['suggested_label'] == 'unknown (nearest: Benign)'


def test_template_report_orders_by_abs_contribution_and_signs_phrases():
    report = generate_template_report(_XAI, _CONTEXT, top_k=2)

    assert 'long flow duration' in report['summary']
    assert 'few SYN flags' in report['summary']
    assert 'f3 high' not in report['summary']
    assert 'Benign' in report['summary']
    assert 'max_softmax=0.550' in report['summary']


def test_template_report_breaks_ties_by_feature_name():
    xai = {'predicted_class': 'A', 'contributions': {'b': 1.0, 'a': -1.0, 'c': 1.0}}

    report = generate_template_report(xai, _CONTEXT, top_k=2)

    assert 'a low' in report['summary']
    assert 'b high' in report['summary']
    assert 'c high' not in report['summary']


def test_template_report_unmapped_feature_uses_fallback():
    xai = {'predicted_class': 'A', 'contributions': {'f0': 0.9, 'f1': -0.4}}

    report = generate_template_report(xai, _CONTEXT)

    assert 'f0 high' in report['summary']
    assert 'f1 low' in report['summary']


def test_template_report_respects_length_limits():
    xai = {
        'predicted_class': 'C' * 300,
        'contributions': {f'very_long_feature_name_{i}': float(i + 1) for i in range(10)},
    }

    report = generate_template_report(xai, _CONTEXT, top_k=10)

    assert 0 < len(report['summary']) <= 200
    assert 0 < len(report['suggested_label']) <= 80
    assert isinstance(report['raw_output'], str)


def test_template_report_factory_returns_template_backend_without_loading_model(monkeypatch):
    def _boom(**kwargs):
        raise AssertionError('from_pretrained must not be called for the template backend')

    monkeypatch.setattr(TransformersLocalBackend, 'from_pretrained', _boom)

    backend = create_local_llm_backend('template', model_name_or_path='ignored', device='cpu')

    assert isinstance(backend, TemplateReportBackend)
    assert 'template' in llm_reporting._BACKEND_TYPES


def test_template_report_factory_still_rejects_unknown_type():
    with pytest.raises(ValueError, match='Unknown local LLM backend type'):
        create_local_llm_backend('nope')


def test_template_report_runtime_stores_three_key_report(tmp_path):
    runtime, clean_chunk = _make_novelty_test_runtime(
        tmp_path, novelty_enabled=True, novelty_tau=0.99, novelty_alpha=0.99
    )
    runtime.llm_backend = TemplateReportBackend()
    _, x_monitor = _score_chunk_novelty(runtime, clean_chunk)
    novelty_flags = np.ones(len(clean_chunk), dtype=bool)

    _generate_novelty_reports(runtime, clean_chunk, x_monitor, novelty_flags)

    assert len(runtime.novelty_reports) > 0
    report = runtime.novelty_reports[0]['llm_report']
    assert set(report) == {'summary', 'suggested_label', 'raw_output'}
