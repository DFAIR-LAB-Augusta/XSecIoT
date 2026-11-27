"""Study B (explanation latency and stability) and Study C (template and LLM reports) for issue #181."""

from __future__ import annotations

import time

from typing import TYPE_CHECKING, Any

import numpy as np

from firce.novelty.evaluation import evaluate_explanation_utility
from firce.novelty.explain import explain_with_shap
from firce.novelty.template_report import generate_template_report
from firce.utils.config import ModelVariant

if TYPE_CHECKING:
    from collections.abc import Sequence

    from firce.novelty.llm_reporting import LocalLLMBackend
    from firce.runtime.sim_types import SimulationRuntime

_TREE_VARIANTS = (ModelVariant.DT, ModelVariant.RF, ModelVariant.XGB)
_KERNEL_BACKGROUND_ROWS = 50


def select_flows_to_explain(
    stream_features: np.ndarray, flags: np.ndarray, probas: np.ndarray, n_samples: int, seed: int
) -> np.ndarray:
    """
    Pick up to n_samples stream row positions to explain.

    A seeded sample of the flagged flows; when none are flagged, the flows with the lowest max-softmax.
    """
    flagged = np.flatnonzero(flags)
    if flagged.size:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(flagged, size=min(n_samples, flagged.size), replace=False))
    return np.argsort(np.max(probas, axis=1), kind='stable')[: min(n_samples, len(stream_features))]


def explain_stream(
    runtime: SimulationRuntime,
    stream_features: np.ndarray,
    flags: np.ndarray,
    feature_names: Sequence[str],
    n_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], np.ndarray]:
    """
    Explain up to n_samples flagged stream flows with SHAP, timing each explanation.

    Uses the tree path for dt/rf/xgb and the kernel path otherwise, mirroring `_explain_model_type`.

    Args:
        runtime: Runtime whose monitor model is explained.
        stream_features: Monitor feature matrix of the stream, row-aligned with `flags`.
        flags: Boolean novelty flags for the stream.
        feature_names: Name of each feature column.
        n_samples: Maximum number of flows to explain.
        seed: Seed for the flow sample and the kernel background.

    Returns:
        The explanations, their latencies in seconds and the explained stream row positions.
    """
    model = runtime.monitor.model
    model_type = 'tree' if runtime.config.model_variant in _TREE_VARIANTS else 'kernel'
    probas = model.predict_proba(stream_features)
    rows = select_flows_to_explain(stream_features, np.asarray(flags, dtype=bool), probas, n_samples, seed)

    background = stream_features
    if model_type == 'kernel' and len(background) > _KERNEL_BACKGROUND_ROWS:
        picked = np.random.default_rng(seed).choice(len(background), size=_KERNEL_BACKGROUND_ROWS, replace=False)
        background = background[picked]

    explanations: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        start = time.perf_counter()
        explanations.append(
            explain_with_shap(model, background, stream_features[row], list(feature_names), model_type=model_type)
        )
        latencies.append(time.perf_counter() - start)
    return explanations, latencies, rows


def run_study_b(
    runtime: SimulationRuntime,
    stream_features: np.ndarray,
    flags: np.ndarray,
    feature_names: Sequence[str],
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Explain flagged stream flows (see `explain_stream`) and return their latency and stability metrics."""
    explanations, latencies, _ = explain_stream(runtime, stream_features, flags, feature_names, n_samples, seed)
    return evaluate_explanation_utility(explanations, latencies)


def run_study_c(
    explanations: Sequence[dict[str, Any]],
    contexts: Sequence[dict[str, float]],
    llm_backend: LocalLLMBackend | None = None,
) -> list[dict[str, Any]]:
    """
    Build a template report per explanation and, with a backend, an LLM report too.

    Args:
        explanations: Outputs of `explain_with_shap`.
        contexts: Novelty context (`max_softmax`, `tau`, `alpha`) per explanation.
        llm_backend: Optional local LLM backend.

    Returns:
        One dict per explanation with `template_report` and `llm_report` (None without a backend).
    """
    from firce.novelty.llm_reporting import generate_report

    records = []
    for explanation, context in zip(explanations, contexts, strict=True):
        records.append({
            'predicted_class': explanation['predicted_class'],
            'template_report': generate_template_report(explanation, context),
            'llm_report': generate_report(llm_backend, explanation, context) if llm_backend is not None else None,
        })
    return records
