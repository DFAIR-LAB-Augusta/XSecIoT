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

import torch

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


def build_report_prompt(xai_result: dict, novelty_context: dict, top_k: int = 5) -> str:
    """
    Build a constrained, structured prompt from #98's XAI output and #97's
    novelty-signal context.

    Implements the proposal's explicit risk mitigation - "constrained LLM
    prompting with structured outputs and bounded vocabulary" - via a fixed
    output-format instruction (Summary: / Suggested label:) that
    parse_report_output expects.

    Only ever includes flow-level numeric features (xai_result['contributions'])
    and novelty-signal scalars (novelty_context) - never raw event payload data,
    per the proposal's governance note.

    Args:
        xai_result: Output of firce.novelty.explain.explain_with_shap/explain_with_lime
            ({'predicted_class': ..., 'contributions': {feature_name: float}}).
        novelty_context: Dict of novelty-signal scalars, e.g. {'max_softmax': ...,
            'tau': ..., 'alpha': ...} (from firce.novelty.decision_rules).
        top_k: Number of top-|contribution| features to include in the prompt.

    Returns:
        A single prompt string ready to pass to a LocalLLMBackend.generate().
    """
    contributions = xai_result['contributions']
    top_features = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    feature_lines = '\n'.join(f'- {name}: {value:+.4f}' for name, value in top_features)

    return (
        'You are a network security assistant. An event was flagged as a possible '
        'unknown/emerging behavior by an automated novelty-detection system.\n\n'
        f"Model's top predicted class: {xai_result['predicted_class']}\n"
        f'Model confidence (max softmax): {novelty_context.get("max_softmax", "unknown")}\n'
        f'Confidence threshold (tau): {novelty_context.get("tau", "unknown")}\n'
        f'Conformal significance (alpha): {novelty_context.get("alpha", "unknown")}\n\n'
        f'Top contributing features (feature: contribution, positive = pushes toward the predicted class):\n'
        f'{feature_lines}\n\n'
        'Respond in exactly this format:\n'
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
