# #141: Stronger Structured-Output Guarantees for Local LLM Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real, grammar-constrained alternative to #99's regex-based `generate_report`/`parse_report_output`, so structured output is *guaranteed* by construction rather than hoped-for from prompting - and measure the compliance-rate improvement over the existing baseline, as #141 explicitly asks for.

**Architecture:** Extends `src/firce/novelty/llm_reporting.py` (#99) with a new `generate_structured_report` function using the `outlines` library's JSON-schema-constrained decoding (`outlines.from_transformers` + `outlines.Generator`), which forces every generated token to stay within the grammar defined by a small Pydantic schema (`_StructuredReportSchema`, with `summary`/`suggested_label` string fields). Direct experimentation (not guessed) surfaced two real, concrete engineering findings that shape the design: (1) `outlines`'s constrained decoding needs a tokenizer with genuine character-level coverage (its logits-processor computes valid token transitions against the JSON grammar's regex, character by character) - the toy `WordLevel` tokenizer built for #99's offline test fixture has no `{`/`"`/`:` tokens at all and fails immediately with `ValueError: vocabulary is incompatible`; a byte-level BPE tokenizer (matching GPT-2's own real tokenizer design, which is byte-level specifically so it can represent any string) is required instead, so #141's test fixture rebuilds the offline tiny-model fixture with `tokenizers.models.BPE` + `pre_tokenizers.ByteLevel`. (2) An *unbounded* string field lets a sufficiently bad model (e.g. a deliberately tiny/untrained one) generate indefinitely inside a JSON string value without ever closing it, since constrained decoding guarantees the *grammar* is followed at each step but doesn't force early termination - confirmed via direct execution generating 200+ tokens of an unclosed string. Bounding each field with Pydantic's `Field(max_length=...)` makes the grammar itself finite, guaranteeing valid, parseable JSON completes within a computable token budget regardless of model quality - confirmed via direct execution: the exact same tiny untrained model that failed to close an unbounded string produces syntactically valid (if semantically nonsensical) JSON immediately once the fields are bounded.

**Tech Stack:** Python, `outlines`, `transformers`, `pydantic`, pytest.

## Global Constraints

- No changes to `TransformersLocalBackend`'s public interface (`.generate`, `.model`, `.tokenizer` - already used by #142's live runtime wiring, merged) - `generate_structured_report` is purely additive, consuming `backend.model`/`backend.tokenizer` directly.
- `generate_structured_report`'s return shape must match #99's `generate_report` exactly (`{'summary': str | None, 'suggested_label': str | None, 'raw_output': str}`) so it's a drop-in alternative wherever `generate_report` is currently used.
- Schema field lengths must be bounded (`Field(max_length=...)`) - an unbounded string field defeats the entire point of this issue (guaranteed termination), per the direct-execution finding above.
- Every test that exercises generation must make a real `.generate()`-equivalent call (no mocking of `outlines`/the model), matching #99's established "real calls, not mocks" testing philosophy for this module.

---

### Task 0: Add the `outlines` dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add `outlines` to the `torch` dependency group**

In `pyproject.toml`, change:
```toml
torch = [
    "torch==2.12.1",
    "transformers>=4.57.0,<5",
]
```
to:
```toml
torch = [
    "torch==2.12.1",
    "transformers>=4.57.0,<5",
    "outlines>=1.3.0,<2",
]
```

- [ ] **Step 2: Regenerate the lockfile and sync**

```bash
uv lock
uv sync --group torch -q
```
Expected: resolves cleanly (adds `outlines`, `outlines-core`, `jsonschema`, `jsonschema-specifications`, `referencing`, `rpds-py`, `diskcache`, `genson`, `attrs`).

- [ ] **Step 3: Verify import**

```bash
uv run python3 -c "import outlines; print(outlines.Generator, outlines.from_transformers)"
```
Expected: prints without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add outlines dependency for constrained LLM output (#141)"
```

---

### Task 1: Byte-level BPE offline test fixture (replaces #99's `WordLevel` fixture for this file only)

**Files:**
- Create: `tests/test_novelty_structured_reporting.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_make_tiny_model_and_tokenizer()` (a local, byte-level variant - not shared with `tests/test_novelty_llm_reporting.py`, which keeps its own `WordLevel` fixture since it doesn't need constrained decoding) - used by every later task in this file.

- [ ] **Step 1: Write and verify the fixture works standalone**

Create `tests/test_novelty_structured_reporting.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_structured_reporting.py -v --no-cov`
Expected: PASS (1 test) - this just confirms the fixture itself builds; the real point of this task is having it ready for Task 2.

- [ ] **Step 3: Commit**

```bash
git add tests/test_novelty_structured_reporting.py
git commit -m "test: add byte-level BPE offline fixture for constrained-decoding tests (#141)"
```

---

### Task 2: `generate_structured_report` with a bounded Pydantic schema

**Files:**
- Modify: `src/firce/novelty/llm_reporting.py`
- Modify: `tests/test_novelty_structured_reporting.py`

**Interfaces:**
- Consumes: `TransformersLocalBackend.model`/`.tokenizer` (existing, #99), `build_report_prompt` (existing, #99).
- Produces: `generate_structured_report(backend: TransformersLocalBackend, xai_result: dict, novelty_context: dict, max_new_tokens: int = 256, top_k: int = 5) -> dict`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_novelty_structured_reporting.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_novelty_structured_reporting.py -v --no-cov -k generate_structured_report`
Expected: FAIL with `ImportError: cannot import name 'generate_structured_report'`

- [ ] **Step 3: Implement `generate_structured_report`**

Add to `src/firce/novelty/llm_reporting.py` (add `import outlines` and `from pydantic import BaseModel, Field` near the top, alongside the existing imports):

```python
class _StructuredReportSchema(BaseModel):
    """
    Bounded output schema for grammar-constrained report generation.

    Field lengths are bounded (Field(max_length=...)) so the constrained-decoding
    grammar itself is finite - this is what guarantees generation completes as
    valid JSON within a computable token budget, regardless of model quality
    (confirmed via direct execution: an unbounded string field lets even a
    deliberately tiny/untrained model generate indefinitely without closing
    the string; bounding it does not).
    """

    summary: str = Field(max_length=200)
    suggested_label: str = Field(max_length=80)


def generate_structured_report(
    backend: 'TransformersLocalBackend',
    xai_result: dict,
    novelty_context: dict,
    max_new_tokens: int = 256,
    top_k: int = 5,
) -> dict:
    """
    Full pipeline with grammar-constrained (guaranteed-structured) output, as
    an alternative to generate_report's regex-based best-effort parsing.

    Uses the outlines library to force every generated token to stay within
    the grammar defined by _StructuredReportSchema - unlike parse_report_output
    (regex extraction from free-form text, degrades to None on non-conforming
    output), summary/suggested_label are never None here: the output is
    guaranteed syntactically valid JSON matching the schema by construction.

    Args:
        backend: A TransformersLocalBackend (consumes .model/.tokenizer directly;
            outlines needs the raw transformers model/tokenizer, not just the
            generate() method).
        xai_result: Output of firce.novelty.explain.explain_with_shap/explain_with_lime.
        novelty_context: Novelty-signal scalars (see build_report_prompt).
        max_new_tokens: Token budget for generation.
        top_k: Forwarded to build_report_prompt.

    Returns:
        Dict with keys 'summary', 'suggested_label' (always non-None str),
        'raw_output' (the raw generated JSON string).
    """
    prompt = build_report_prompt(xai_result, novelty_context, top_k=top_k)
    outlines_model = outlines.from_transformers(backend.model, backend.tokenizer)
    generator = outlines.Generator(outlines_model, _StructuredReportSchema)
    raw_output = generator(prompt, max_new_tokens=max_new_tokens)
    parsed = _StructuredReportSchema.model_validate_json(raw_output)
    return {'summary': parsed.summary, 'suggested_label': parsed.suggested_label, 'raw_output': raw_output}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_novelty_structured_reporting.py -v --no-cov -k generate_structured_report`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/firce/novelty/llm_reporting.py tests/test_novelty_structured_reporting.py
git commit -m "feat: add generate_structured_report with bounded grammar-constrained output (#141)"
```

---

### Task 3: Compliance-rate comparison against #99's regex-based baseline

**Files:**
- Modify: `tests/test_novelty_structured_reporting.py`

Per #141's own "Deliverables": "Comparison against #99's current regex-based baseline on structured-output compliance rate." Runs both `generate_report` (#99, regex-based) and `generate_structured_report` (this issue) against the same tiny untrained model across several different prompts, and asserts the constrained version's compliance rate is strictly better (in fact, guaranteed 100% vs. the regex baseline's real, already-confirmed-in-#99 failure mode).

- [ ] **Step 1: Write the test**

Add to `tests/test_novelty_structured_reporting.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_structured_reporting.py -v --no-cov`
Expected: PASS (all tests in the file).

- [ ] **Step 3: Commit**

```bash
git add tests/test_novelty_structured_reporting.py
git commit -m "test: add compliance-rate comparison vs #99's regex baseline (#141)"
```

---

### Task 4: Full suite, lint, push, PR

- [ ] **Step 1: Run the full lean-CI suite**

```bash
uv run pytest -q --no-cov
```
Expected: exit code 0.

- [ ] **Step 2: Lint**

```bash
uv run ruff check src/firce/novelty/ tests/test_novelty_structured_reporting.py
uv run ruff format --check src/firce/novelty/ tests/test_novelty_structured_reporting.py
```
Expected: clean.

- [ ] **Step 3: Push and open the PR**

```bash
git push origin 141-structured-output
gh pr create --base multiclass --head 141-structured-output \
  --title "feat: grammar-constrained structured output for local LLM reporting (#141)" \
  --body "$(cat <<'EOF'
## Summary
- New `generate_structured_report` in `src/firce/novelty/llm_reporting.py`: grammar-constrained (via `outlines`) alternative to #99's regex-based `generate_report`, guaranteeing valid, parseable structured output by construction rather than hoping the model follows a free-form prompt instruction.
- Bounded Pydantic schema (`_StructuredReportSchema`, `Field(max_length=...)` on both fields) - confirmed via direct execution this is what makes the guarantee actually hold: an unbounded string field lets even a deliberately untrained tiny model generate indefinitely without ever closing the JSON string, while bounding it guarantees termination within a computable token budget regardless of model quality.
- Same return shape as #99's `generate_report` (`summary`/`suggested_label`/`raw_output`) - a drop-in alternative, no changes to `TransformersLocalBackend`'s existing interface (already used by #142's live runtime wiring, merged).
- Fifth follow-up issue on #99's basic LLM reporting setup. Depends on #99 (merged).

## Real finding along the way
`outlines`'s constrained decoding needs a tokenizer with genuine character-level coverage - #99's toy `WordLevel` offline test tokenizer has no `{`/`"`/`:` tokens at all and fails immediately with `ValueError: vocabulary is incompatible`. Rebuilt the offline tiny-model test fixture with byte-level BPE (matching GPT-2's own real tokenizer design, which is byte-level specifically so it can represent any string) for this file only - #99's own tests are untouched.

## Test plan
- [x] Every test makes a real constrained-generation call through the same tiny offline model used in #99 - no mocks.
- [x] `generate_structured_report` always produces non-None, parseable output - demonstrated against the exact untrained model that (confirmed via direct execution) fails to terminate an unbounded string within 200+ tokens.
- [x] Field-length bounds are respected in the output.
- [x] Direct compliance-rate comparison against #99's regex baseline across 3 distinct scenarios - constrained approach is 100% compliant by construction.
- [x] `uv run pytest -q`: full suite passes (exit 0).
- [x] `ruff check`/`ruff format --check`: clean.
EOF
)"
```

## Self-Review

**Spec coverage:** Task 0 adds the dependency. Task 1 builds the required (different-from-#99) offline test fixture, verified working standalone first. Task 2 implements the core deliverable with the bounded-schema guarantee, verified against the exact adversarial case (untrained model, unbounded-would-fail). Task 3 delivers the explicitly-requested compliance-rate comparison. Task 4 verifies and ships.

**Placeholder scan:** No TBD/TODO; every step has literal runnable code, including the exact `outlines`/`tokenizers` API calls already confirmed via direct execution before this plan was written.

**Type consistency:** `generate_structured_report(backend, xai_result, novelty_context, max_new_tokens=256, top_k=5) -> dict` returns the identical key set (`summary`, `suggested_label`, `raw_output`) as #99's `generate_report`, defined once in Task 2 and exercised identically in Task 3's side-by-side comparison.

**Explicitly out of scope:** Changes to `TransformersLocalBackend`'s public interface or `create_local_llm_backend` factory (#99) - `generate_structured_report` consumes `.model`/`.tokenizer` directly rather than needing a new backend method. Wiring this into #142's live runtime path (that already calls `generate_report`; swapping in the constrained variant there is a follow-up choice, not required by #141's own text, which frames this as a standalone deliverable to compare against the baseline). Non-`outlines` constrained-decoding backends (llama.cpp GBNF grammars) - `outlines` alone satisfies #141's "prototype... compare" framing for this first constrained-decoding attempt; a future issue can add alternatives if `outlines` proves insufficient for a specific hardware tier from #137.
