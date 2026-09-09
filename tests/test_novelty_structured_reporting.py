from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


_CORPUS = (
    'novelty detected class benign portscan xmasattack summary label suggested feature '
    'contributes value increase decrease scanning burst possible unknown emerging behavior '
    '{"summary": "example text here", "suggested_label": "possible scanning burst"}'
)


def _make_tiny_tokenizer():
    # outlines' constrained decoding needs a tokenizer with genuine character-level
    # coverage (its logits processor walks the JSON grammar character by character) -
    # confirmed via direct execution that #99's WordLevel fixture tokenizer fails
    # immediately with "vocabulary is incompatible" since it has no '{'/'"'/':' tokens
    # at all. Byte-level BPE (matching GPT-2's own real tokenizer design) can
    # represent any string, so it works regardless of training corpus content.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=['[PAD]', '[BOS]', '[EOS]'])
    tokenizer.train_from_iterator([_CORPUS], trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')


def _make_tiny_model_and_tokenizer():
    tokenizer = _make_tiny_tokenizer()
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        # 768: confirmed via direct execution that build_report_prompt's real
        # output is ~360 tokens with this byte-level BPE tokenizer (fragments
        # much more heavily than #99's WordLevel tokenizer at the same tiny
        # vocab size), plus room for a 200-256 token generation budget.
        n_positions=768,
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


def test_tiny_fixture_builds_without_error():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    assert model is not None
    assert tokenizer.vocab_size > 0


from firce.novelty.llm_reporting import TransformersLocalBackend, generate_structured_report


def test_generate_structured_report_always_produces_parseable_output():
    # The exact tiny UNTRAINED model that (per #141's own investigation) fails
    # to close an unbounded JSON string within 200+ tokens produces valid,
    # parseable output here once the schema's fields are bounded - this is
    # the guarantee this issue is about, demonstrated against the worst case
    # (a model with no learned tendency to produce sensible output at all).
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    xai_result = {
        'predicted_class': 'Benign',
        'contributions': {'flow_duration': 0.8, 'tot_fwd_pkt': -0.3},
    }
    novelty_context = {'max_softmax': 0.42, 'tau': 0.6, 'alpha': 0.3}

    report = generate_structured_report(backend, xai_result, novelty_context, max_new_tokens=200)

    # Unlike #99's generate_report, these must never be None - that's the
    # entire point of grammar-constrained decoding: guaranteed structure.
    assert report['summary'] is not None
    assert report['suggested_label'] is not None
    assert isinstance(report['summary'], str)
    assert isinstance(report['suggested_label'], str)
    assert isinstance(report['raw_output'], str)


def test_generate_structured_report_respects_field_length_bounds():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    xai_result = {'predicted_class': 'Benign', 'contributions': {'flow_duration': 0.5}}
    novelty_context = {'max_softmax': 0.5, 'tau': 0.6, 'alpha': 0.3}

    report = generate_structured_report(backend, xai_result, novelty_context, max_new_tokens=200)

    assert len(report['summary']) <= 200
    assert len(report['suggested_label']) <= 80


from firce.novelty.llm_reporting import generate_report


def test_structured_output_compliance_rate_beats_regex_baseline():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    # Several distinct XAI inputs, to avoid drawing a conclusion from a single
    # lucky/unlucky generation.
    scenarios = [
        {'predicted_class': 'Benign', 'contributions': {'flow_duration': 0.8}},
        {'predicted_class': 'PortScan', 'contributions': {'tot_fwd_pkt': -0.5}},
        {'predicted_class': 'XMasAttack', 'contributions': {'totlen_fwd_pkts': 0.3}},
    ]
    novelty_context = {'max_softmax': 0.5, 'tau': 0.6, 'alpha': 0.3}

    regex_compliant = 0
    structured_compliant = 0
    for xai_result in scenarios:
        regex_report = generate_report(backend, xai_result, novelty_context, max_new_tokens=15)
        if regex_report['summary'] is not None and regex_report['suggested_label'] is not None:
            regex_compliant += 1

        structured_report = generate_structured_report(backend, xai_result, novelty_context, max_new_tokens=200)
        if structured_report['summary'] is not None and structured_report['suggested_label'] is not None:
            structured_compliant += 1

    # The constrained approach is guaranteed compliant by construction; the
    # regex baseline against this untrained model is not (confirmed in #99:
    # it can produce non-conforming free-form text with no Summary:/Suggested
    # label: lines at all).
    assert structured_compliant == len(scenarios)
    assert structured_compliant >= regex_compliant
