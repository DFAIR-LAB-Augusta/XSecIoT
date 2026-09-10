from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from firce.novelty.llm_reporting import TransformersLocalBackend, evaluate_prompt_strategies

_CORPUS = (
    'novelty detected class benign portscan xmasattack summary label suggested feature '
    'contributes value increase decrease scanning burst possible unknown emerging behavior '
    'reasoning example analyze first step'
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
        n_positions=512,  # few-shot prompts are the longest of the three strategies
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


def test_evaluate_prompt_strategies_returns_compliance_rate_per_strategy():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    scenarios = [
        {'predicted_class': 'Benign', 'contributions': {'flow_duration': 0.8}},
        {'predicted_class': 'PortScan', 'contributions': {'tot_fwd_pkt': -0.5}},
    ]
    novelty_context = {'max_softmax': 0.5, 'tau': 0.6, 'alpha': 0.3}

    results = evaluate_prompt_strategies(backend, scenarios, novelty_context, max_new_tokens=20)

    assert set(results.keys()) == {'zero_shot', 'few_shot', 'cot'}
    for strategy_result in results.values():
        assert 0.0 <= strategy_result['compliance_rate'] <= 1.0
        assert len(strategy_result['reports']) == len(scenarios)


def test_evaluate_prompt_strategies_respects_strategies_subset():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    backend = TransformersLocalBackend(model, tokenizer)

    scenarios = [{'predicted_class': 'Benign', 'contributions': {'flow_duration': 0.8}}]
    novelty_context = {'max_softmax': 0.5, 'tau': 0.6, 'alpha': 0.3}

    results = evaluate_prompt_strategies(
        backend, scenarios, novelty_context, strategies=('zero_shot',), max_new_tokens=20
    )

    assert set(results.keys()) == {'zero_shot'}
