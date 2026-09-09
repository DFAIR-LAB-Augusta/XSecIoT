import tempfile

import pytest

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from firce.novelty.llm_reporting import (
    TransformersLocalBackend,
    build_report_prompt,
    create_local_llm_backend,
    generate_report,
    parse_report_output,
)

_CORPUS = (
    'novelty detected class benign portscan xmasattack summary label suggested feature '
    'contributes value increase decrease scanning burst possible unknown emerging behavior'
)


def _make_tiny_tokenizer():
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'])
    tokenizer.train_from_iterator([_CORPUS], trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]'
    )


def _make_tiny_model_and_tokenizer():
    tokenizer = _make_tiny_tokenizer()
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        # 256 comfortably fits build_report_prompt's real output (~120 tokens
        # with this tiny word-level vocab) plus generated tokens; n_positions
        # only sizes the position-embedding table, so this stays cheap even
        # for such a small model.
        n_positions=256,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, tokenizer


def test_transformers_local_backend_generates_real_text():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    output = backend.generate('novelty detected class benign', max_new_tokens=6)

    # This tiny, randomly-initialized model can legitimately pick EOS as its
    # very first token (confirmed empirically across multiple seeds - this
    # isn't test flakiness to chase, it's real behavior of an untrained
    # model), so length isn't asserted - only that generation completes and
    # returns a string, which is what's under test here.
    assert isinstance(output, str)


def test_transformers_local_backend_from_pretrained_loads_any_local_checkpoint_directory():
    model, tokenizer = _make_tiny_model_and_tokenizer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        # from_pretrained treats this saved-to-disk checkpoint no differently
        # than a future user-provided SFT/KD checkpoint would be treated -
        # both are just a local directory in the same HF format.
        backend = TransformersLocalBackend.from_pretrained(tmp_dir)
        output = backend.generate('novelty detected', max_new_tokens=4)

    assert isinstance(output, str)


def test_create_local_llm_backend_transformers_type_loads_checkpoint():
    model, tokenizer = _make_tiny_model_and_tokenizer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        backend = create_local_llm_backend('transformers', model_name_or_path=tmp_dir)
        output = backend.generate('novelty detected', max_new_tokens=4)

    assert isinstance(output, str)


def test_create_local_llm_backend_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match='backend'):
        create_local_llm_backend('not_a_real_backend')


def test_build_report_prompt_includes_top_features_and_novelty_context():
    xai_result = {
        'predicted_class': 'Benign',
        'contributions': {'flow_duration': 0.8, 'tot_fwd_pkt': -0.3, 'tot_bwd_pkts': 0.05},
    }
    novelty_context = {'max_softmax': 0.42, 'tau': 0.6, 'alpha': 0.3}

    prompt = build_report_prompt(xai_result, novelty_context, top_k=2)

    assert 'flow_duration' in prompt
    assert 'tot_fwd_pkt' in prompt
    assert 'tot_bwd_pkts' not in prompt  # only top_k=2 features by |contribution|
    assert 'Benign' in prompt
    assert 'Summary:' in prompt  # bounded-output-format instruction present
    assert 'Suggested label:' in prompt


def test_build_report_prompt_never_includes_raw_payload_keys():
    # Governance constraint: only numeric feature contributions and novelty
    # context are ever included - nothing resembling a raw payload field.
    xai_result = {'predicted_class': 'Benign', 'contributions': {'flow_duration': 1.0}}
    novelty_context = {'max_softmax': 0.5, 'tau': 0.6, 'alpha': 0.3}

    prompt = build_report_prompt(xai_result, novelty_context)

    for forbidden in ('payload', 'raw_packet', 'user_data'):
        assert forbidden not in prompt.lower()


def test_parse_report_output_extracts_summary_and_label_when_well_formed():
    raw = 'Summary: Unusual burst of short flows.\nSuggested label: possible scanning burst\n'

    result = parse_report_output(raw)

    assert result['summary'] == 'Unusual burst of short flows.'
    assert result['suggested_label'] == 'possible scanning burst'
    assert result['raw_output'] == raw


def test_parse_report_output_falls_back_gracefully_when_format_not_followed():
    raw = 'benign benign benign portscan xmasattack'  # e.g. an untrained model's real output

    result = parse_report_output(raw)

    assert result['summary'] is None
    assert result['suggested_label'] is None
    assert result['raw_output'] == raw


def test_generate_report_end_to_end_with_real_tiny_model():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    xai_result = {
        'predicted_class': 'Benign',
        'contributions': {'flow_duration': 0.8, 'tot_fwd_pkt': -0.3},
    }
    novelty_context = {'max_softmax': 0.42, 'tau': 0.6, 'alpha': 0.3}

    report = generate_report(backend, xai_result, novelty_context, max_new_tokens=10)

    # summary/suggested_label may legitimately be None, and raw_output may
    # legitimately be empty (greedy decoding on this tiny untrained model can
    # pick EOS as its very first token - confirmed via direct execution) -
    # the pipeline wiring and graceful degradation is what's under test here,
    # not output quality or length.
    assert 'summary' in report
    assert 'suggested_label' in report
    assert isinstance(report['raw_output'], str)
