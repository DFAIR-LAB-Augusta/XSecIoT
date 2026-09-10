# firce.novelty.llm_reporting
"""
LLM-assisted reporting & candidate class naming for flagged unknown/emerging
behaviors (dissertation proposal Section 4.4.4).

Runs locally (not a hosted API) - a reporting/suggestion tool, not a
ground-truth labeling oracle, per the proposal. The backend is engineered
against a variety of local checkpoints, not one hardcoded model: any local
directory in HF format loads identically via TransformersLocalBackend.from_pretrained,
whether it's a stock pretrained checkpoint or a future user-trained SFT/KD
checkpoint (confirmed via direct save_pretrained/from_pretrained round-trip -
no special-casing needed for a fine-tuned model).

Governance note (per proposal): only flow-level numeric features and
structured XAI attributions (firce.novelty.explain output) are ever included
in prompts - never raw event payload data.
"""

import logging
import re

from typing import Any, Optional, Protocol

import outlines
import torch

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LocalLLMBackend(Protocol):
    """Protocol for any local LLM inference backend used by this module."""

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate text continuing from `prompt`, returning only the newly generated text."""
        ...


class TransformersLocalBackend:
    """
    LocalLLMBackend implementation backed by the transformers library.

    Works with any causal-LM checkpoint transformers.AutoModelForCausalLM can
    load - construct directly with an already-loaded model/tokenizer pair
    (useful for tests or programmatic model construction), or via
    from_pretrained(path) for any local checkpoint directory.
    """

    def __init__(self, model: Any, tokenizer: Any, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: Optional[str] = None) -> 'TransformersLocalBackend':
        """
        Load any local checkpoint directory (or hub model name) in HF format.

        A future user-trained SFT/KD checkpoint saved via model.save_pretrained(path) /
        tokenizer.save_pretrained(path) loads through this exact same path, with
        no code changes needed here.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, device=device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_ids = output_ids[0][inputs['input_ids'].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


_BACKEND_TYPES = ('transformers',)


def create_local_llm_backend(backend_type: str, **kwargs) -> LocalLLMBackend:
    """
    Construct a LocalLLMBackend by type name.

    Currently supports 'transformers' (TransformersLocalBackend.from_pretrained).
    Kept as an explicit factory (rather than calling TransformersLocalBackend
    directly) so additional local-inference backends (e.g. llama.cpp, vllm,
    ollama) can be added later without changing call sites - "variety of
    models" is a deployment/engineering decision this factory keeps open,
    not something this issue commits to a single implementation of.

    Args:
        backend_type: One of the supported backend type names.
        **kwargs: Forwarded to the backend's construction (for 'transformers',
            this is TransformersLocalBackend.from_pretrained's kwargs:
            model_name_or_path, device).

    Raises:
        ValueError: If backend_type is not a supported backend.
    """
    if backend_type == 'transformers':
        return TransformersLocalBackend.from_pretrained(**kwargs)
    raise ValueError(f'Unknown local LLM backend type {backend_type!r}; expected one of {_BACKEND_TYPES}')


_FEW_SHOT_EXAMPLES = (
    (
        {'predicted_class': 'Benign', 'contributions': {'flow_duration': 1.2, 'tot_fwd_pkt': -0.1}},
        {'max_softmax': 0.55, 'tau': 0.6, 'alpha': 0.3},
        'Summary: A single flow with an unusually long duration but otherwise typical packet counts.\n'
        'Suggested label: possible long-lived idle connection\n',
    ),
    (
        {'predicted_class': 'PortScan', 'contributions': {'tot_fwd_pkt': 2.1, 'flow_duration': -0.9}},
        {'max_softmax': 0.48, 'tau': 0.6, 'alpha': 0.3},
        'Summary: A burst of short flows with high forward-packet counts targeting multiple ports.\n'
        'Suggested label: possible scanning burst\n',
    ),
)

_VALID_STRATEGIES = ('zero_shot', 'few_shot', 'cot')


def _format_feature_lines(xai_result: dict, top_k: int) -> str:
    contributions = xai_result['contributions']
    top_features = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    return '\n'.join(f'- {name}: {value:+.4f}' for name, value in top_features)


def _format_context_block(xai_result: dict, novelty_context: dict, top_k: int) -> str:
    feature_lines = _format_feature_lines(xai_result, top_k)
    return (
        f"Model's top predicted class: {xai_result['predicted_class']}\n"
        f'Model confidence (max softmax): {novelty_context.get("max_softmax", "unknown")}\n'
        f'Confidence threshold (tau): {novelty_context.get("tau", "unknown")}\n'
        f'Conformal significance (alpha): {novelty_context.get("alpha", "unknown")}\n\n'
        f'Top contributing features (feature: contribution, positive = pushes toward the predicted class):\n'
        f'{feature_lines}\n\n'
    )


def build_report_prompt(xai_result: dict, novelty_context: dict, top_k: int = 5, strategy: str = 'zero_shot') -> str:
    """
    Build a constrained, structured prompt from #98's XAI output and #97's
    novelty-signal context.

    Implements the proposal's explicit risk mitigation - "constrained LLM
    prompting with structured outputs and bounded vocabulary" - via a fixed
    output-format instruction (Summary: / Suggested label:) that
    parse_report_output expects, identical across all three strategies.

    Only ever includes flow-level numeric features (xai_result['contributions'])
    and novelty-signal scalars (novelty_context) - never raw event payload data,
    per the proposal's governance note.

    Three prototyped strategies (#138, "prototype and compare" per the
    proposal - not a single committed design):
      - 'zero_shot' (default): #99's original single-shot template, unchanged.
      - 'few_shot': prepends two small worked examples before the actual query.
      - 'cot': adds a "reason step by step first" instruction and a Reasoning:
        scratchpad line before the final Summary:/Suggested label: answer.

    Args:
        xai_result: Output of firce.novelty.explain.explain_with_shap/explain_with_lime
            ({'predicted_class': ..., 'contributions': {feature_name: float}}).
        novelty_context: Dict of novelty-signal scalars, e.g. {'max_softmax': ...,
            'tau': ..., 'alpha': ...} (from firce.novelty.decision_rules).
        top_k: Number of top-|contribution| features to include in the prompt.
        strategy: One of 'zero_shot', 'few_shot', 'cot'.

    Returns:
        A single prompt string ready to pass to a LocalLLMBackend.generate().

    Raises:
        ValueError: If strategy is not a supported value.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f'Unknown prompt strategy {strategy!r}; expected one of {_VALID_STRATEGIES}')

    intro = (
        'You are a network security assistant. An event was flagged as a possible '
        'unknown/emerging behavior by an automated novelty-detection system.\n\n'
    )
    context_block = _format_context_block(xai_result, novelty_context, top_k)

    if strategy == 'few_shot':
        examples_block = ''
        for example_xai, example_context, example_answer in _FEW_SHOT_EXAMPLES:
            examples_block += (
                'Example:\n' + _format_context_block(example_xai, example_context, top_k) + example_answer + '\n'
            )
        return (
            intro
            + 'Here are two worked examples of the expected response format:\n\n'
            + examples_block
            + 'Now analyze this new event:\n\n'
            + context_block
            + 'Respond in exactly this format:\n'
            'Summary: <one sentence describing the anomalous behavior>\n'
            'Suggested label: <a short descriptive phrase, not a definitive classification>\n'
        )

    if strategy == 'cot':
        return (
            intro
            + context_block
            + 'First, reason step by step about what these feature contributions suggest, writing your '
            'reasoning after "Reasoning:". Then give your final answer in exactly this format:\n'
            'Reasoning: <your step-by-step analysis>\n'
            'Summary: <one sentence describing the anomalous behavior>\n'
            'Suggested label: <a short descriptive phrase, not a definitive classification>\n'
        )

    # strategy == 'zero_shot' (#99's original template, unchanged)
    return (
        intro + context_block + 'Respond in exactly this format:\n'
        'Summary: <one sentence describing the anomalous behavior>\n'
        'Suggested label: <a short descriptive phrase, not a definitive classification>\n'
    )


_SUMMARY_PATTERN = re.compile(r'Summary:\s*(.+)')
_LABEL_PATTERN = re.compile(r'Suggested label:\s*(.+)')


def parse_report_output(raw_text: str) -> dict:
    """
    Parse LLM output against the constrained Summary:/Suggested label: format
    build_report_prompt requests.

    Degrades gracefully (returns None for fields it can't find) rather than
    raising - real generation, especially from a small/untrained model or any
    model that doesn't follow instructions perfectly, will not always match
    the requested format exactly.

    Args:
        raw_text: The LLM backend's raw generated text.

    Returns:
        Dict with keys 'summary', 'suggested_label' (either a str or None),
        and 'raw_output' (always the original raw_text, for debugging/audit).
    """
    summary_match = _SUMMARY_PATTERN.search(raw_text)
    label_match = _LABEL_PATTERN.search(raw_text)
    return {
        'summary': summary_match.group(1).strip() if summary_match else None,
        'suggested_label': label_match.group(1).strip() if label_match else None,
        'raw_output': raw_text,
    }


def generate_report(
    backend: LocalLLMBackend,
    xai_result: dict,
    novelty_context: dict,
    max_new_tokens: int = 128,
    top_k: int = 5,
) -> dict:
    """
    Full pipeline: build a constrained prompt from #98's XAI output + #97's
    novelty context, generate via a local LLM backend, and parse the result.

    Args:
        backend: Any LocalLLMBackend (e.g. TransformersLocalBackend).
        xai_result: Output of firce.novelty.explain.explain_with_shap/explain_with_lime.
        novelty_context: Novelty-signal scalars (see build_report_prompt).
        max_new_tokens: Forwarded to backend.generate.
        top_k: Forwarded to build_report_prompt.

    Returns:
        Dict with keys 'summary', 'suggested_label' (str or None), 'raw_output' (str).
    """
    prompt = build_report_prompt(xai_result, novelty_context, top_k=top_k)
    raw_output = backend.generate(prompt, max_new_tokens=max_new_tokens)
    return parse_report_output(raw_output)


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


def evaluate_prompt_strategies(
    backend: 'TransformersLocalBackend',
    scenarios: list,
    novelty_context: dict,
    strategies: tuple = _VALID_STRATEGIES,
    max_new_tokens: int = 64,
) -> dict:
    """
    Compare prompt strategies on structured-output compliance rate (#138's
    explicit deliverable), using #99's regex-based generate_report/
    parse_report_output - not #141's grammar-constrained generate_structured_report,
    which is always 100% compliant by construction regardless of prompt content
    and would be a useless differentiator for this specific comparison.

    Results reflect whatever local model backend is passed in - a small,
    untrained test model will show different (likely less meaningful)
    behavior than a real trained checkpoint; this harness measures the
    comparison mechanism, not a universal claim about which strategy is best.

    Args:
        backend: A TransformersLocalBackend.
        scenarios: List of #98's explain_with_shap/explain_with_lime output dicts.
        novelty_context: Novelty-signal scalars shared across all scenarios (see build_report_prompt).
        strategies: Which strategies to compare (subset of _VALID_STRATEGIES).
        max_new_tokens: Forwarded to generate_report.

    Returns:
        Dict mapping each strategy to {'compliance_rate': float, 'reports': list[dict]}
        (reports are generate_report's raw per-scenario output, for inspection).
    """
    results = {}
    for strategy in strategies:
        reports = []
        for xai_result in scenarios:
            prompt = build_report_prompt(xai_result, novelty_context, strategy=strategy)
            raw_output = backend.generate(prompt, max_new_tokens=max_new_tokens)
            reports.append(parse_report_output(raw_output))

        compliant = sum(1 for r in reports if r['summary'] is not None and r['suggested_label'] is not None)
        results[strategy] = {
            'compliance_rate': compliant / len(scenarios) if scenarios else 0.0,
            'reports': reports,
        }
    return results
