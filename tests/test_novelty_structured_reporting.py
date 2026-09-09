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


def test_tiny_fixture_builds_without_error():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    assert model is not None
    assert tokenizer.vocab_size > 0
