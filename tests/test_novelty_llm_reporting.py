import tempfile

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from firce.novelty.llm_reporting import TransformersLocalBackend


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
        n_positions=64,
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

    assert isinstance(output, str)
    assert len(output) > 0


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
    assert len(output) > 0
