"""Direction 3 experiment driver: Study A over seeds, with optional Studies B and C (issue #181)."""

from __future__ import annotations

import dataclasses
import json
import subprocess

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from firce.experiments.scoring import build_runtime, score_frame
from firce.experiments.split import split_device_time_blocks
from firce.experiments.study_a import build_scores_frame, evaluate_seed, make_run_id, write_results
from firce.experiments.study_bc import explain_stream, run_study_c
from firce.novelty.evaluation import evaluate_explanation_utility

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from firce.experiments.presets import DatasetPreset
    from firce.runtime.sim_types import SimulationRuntime

BC_RULE = 'combined'
BC_TARGET = 0.05


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'nogit'


def class_names_of(runtime: SimulationRuntime) -> list[str]:
    """Class name of each probability column, decoding integer-encoded `classes_` via the label encoder."""
    classes = np.asarray(runtime.monitor.model.classes_)
    if runtime.label_encoder is not None and classes.dtype.kind in 'iu':
        classes = runtime.label_encoder.inverse_transform(classes)
    return [str(c) for c in classes]


def _feature_names(runtime: SimulationRuntime, n_features: int) -> list[str]:
    names = getattr(runtime.scaler, 'feature_names_in_', None)
    if names is not None and len(names) == n_features:
        return [str(n) for n in names]
    return [f'f{i}' for i in range(n_features)]


def _study_bc(
    runtime: SimulationRuntime,
    stream: Any,
    flags: np.ndarray,
    row: dict[str, Any],
    class_names: Sequence[str],
    seed_dir: Path,
    seed: int,
    *,
    study_b: bool,
    study_c: bool,
    n_samples: int,
    llm_backend: Any,
) -> None:
    explanations, latencies, rows = explain_stream(
        runtime, stream.features, flags, _feature_names(runtime, stream.features.shape[1]), n_samples, seed
    )
    by_code = {str(c): name for c, name in zip(runtime.monitor.model.classes_, class_names, strict=True)}
    for explanation in explanations:
        explanation['predicted_class'] = by_code[str(explanation['predicted_class'])]
    if study_b:
        utility = evaluate_explanation_utility(explanations, latencies)
        utility['n_explained'] = len(explanations)
        _dump_json(seed_dir / 'study_b.json', utility)
    if study_c:
        max_softmax = np.max(stream.probas, axis=1)
        contexts = [
            {'max_softmax': float(max_softmax[i]), 'tau': float(row['tau']), 'alpha': float(row['alpha'])} for i in rows
        ]
        reports = run_study_c(explanations, contexts, llm_backend)
        lines = [
            json.dumps({'row_index': int(i), **report}, default=_json_default)
            for i, report in zip(rows, reports, strict=True)
        ]
        (seed_dir / 'study_c.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''))


def run_experiment(
    preset: DatasetPreset,
    seeds: Sequence[int],
    out_root: Path,
    *,
    held_out: Sequence[str] | None = None,
    model_variant: str = 'rf',
    ce_type: str = 'ice',
    use_mlp: bool = False,
    study_b: bool = False,
    study_c: bool = False,
    study_b_samples: int = 50,
    llm_backend_type: str | None = None,
    llm_model_path: str | None = None,
) -> Path:
    """
    Run Study A over `seeds` (plus Studies B and C when requested) and write the results directory.

    Args:
        preset: Dataset preset (path, columns, default held-out classes).
        seeds: One split-and-model seed per run.
        out_root: Root directory; results go to `out_root/<preset.name>/<run id>/`.
        held_out: Held-out class names overriding `preset.held_out`.
        model_variant: Classifier variant name.
        ce_type: Conformal evaluator type name.
        use_mlp: Use the multiclass MLP as the conformal monitor model.
        study_b: Also explain flagged flows and write `seed_<k>/study_b.json`.
        study_c: Also write template reports to `seed_<k>/study_c.jsonl`.
        study_b_samples: Maximum flows explained per seed.
        llm_backend_type: Local LLM backend type for Study C reports.
        llm_model_path: Model name or path for that backend.

    Returns:
        The run directory.
    """
    if held_out is not None:
        preset = dataclasses.replace(preset, held_out=tuple(held_out))
    run_dir = out_root / preset.name / make_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_backend = None
    if study_c and llm_backend_type is not None:
        from firce.novelty.llm_reporting import create_local_llm_backend

        llm_backend = create_local_llm_backend(llm_backend_type, model_name_or_path=llm_model_path)

    data = pd.read_csv(preset.resolved_path)
    per_seed: dict[int, list[dict[str, Any]]] = {}
    scores: dict[int, pd.DataFrame] = {}
    val_strategies: dict[int, str] = {}
    for seed in seeds:
        split = split_device_time_blocks(data, preset, seed)
        val_strategies[seed] = split.val_strategy
        seed_dir = run_dir / f'seed_{seed}'
        runtime = build_runtime(split.train, preset, seed_dir / 'work', seed, model_variant, ce_type, use_mlp)
        val = score_frame(runtime, split.val, preset)
        stream = score_frame(runtime, split.stream, preset)
        names = class_names_of(runtime)
        rows = evaluate_seed(val, stream, preset.held_out, names)
        per_seed[seed] = rows
        scores[seed] = build_scores_frame(stream, rows)
        if study_b or study_c:
            row = next(r for r in rows if r['rule'] == BC_RULE and r['target_fa'] == BC_TARGET)
            flags = scores[seed][f'flag_{BC_RULE}_{BC_TARGET}'].to_numpy(dtype=bool)
            seed_dir.mkdir(parents=True, exist_ok=True)
            _study_bc(
                runtime,
                stream,
                flags,
                row,
                names,
                seed_dir,
                seed,
                study_b=study_b,
                study_c=study_c,
                n_samples=study_b_samples,
                llm_backend=llm_backend,
            )

    config = {
        'preset': preset.name,
        'data_path': str(preset.resolved_path),
        'seeds': list(seeds),
        'held_out': list(preset.held_out),
        'model_variant': model_variant,
        'ce_type': ce_type,
        'use_mlp': use_mlp,
        'study_b': study_b,
        'study_b_samples': study_b_samples,
        'study_c': study_c,
        'llm_backend_type': llm_backend_type,
        'llm_model_path': llm_model_path,
        'git_sha': _git_sha(),
        'val_strategy': {str(seed): strategy for seed, strategy in val_strategies.items()},
    }
    write_results(run_dir, config, per_seed, scores)
    return run_dir
