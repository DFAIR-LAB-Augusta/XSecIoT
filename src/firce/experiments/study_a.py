"""
Study A evaluation: scored splits to per-seed metrics and a cross-seed summary (issue #181).

For each seed, rule and matched false-alert target, thresholds are chosen on the
validation split and applied to the stream. Open-world, closed-world and session
delay metrics are then computed and written under a per-run results directory.

Timestamps on DFAIR have minute resolution (risk R3, accepted), so delays are
multiples of 60 seconds and flows sharing a minute are tied in time order.
"""

from __future__ import annotations

import json
import subprocess

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from firce.novelty.decision_rules import NOVELTY_RULES, apply_novelty_rule
from firce.novelty.evaluation import evaluate_closed_world, evaluate_detection_delay, evaluate_open_world_novelty
from firce.novelty.thresholds import DEFAULT_FA_TARGETS, select_thresholds

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

    from firce.experiments.scoring import ScoredSplit

DEFAULT_LOOKBACK_S = 1800
_KEY_COLUMNS = ('rule', 'target_fa')


def _max_p(split: ScoredSplit) -> np.ndarray:
    return np.max(np.stack(list(split.p_values.values()), axis=1), axis=1)


def novelty_scores(rule: str, split: ScoredSplit) -> np.ndarray:
    """
    Continuous novelty score (higher = more novel) for AUROC/AUPRC.

    1 - max_softmax for `confidence`, 1 - max p-value for `conformal` and
    1 - max(max_softmax, max p-value) for `combined`.
    """
    max_softmax = np.max(split.probas, axis=1)
    if rule == 'confidence':
        return 1.0 - max_softmax
    if rule == 'conformal':
        return 1.0 - _max_p(split)
    if rule == 'combined':
        return 1.0 - np.maximum(max_softmax, _max_p(split))
    raise ValueError(f'Unknown novelty rule {rule!r}; expected one of {NOVELTY_RULES}')


def build_sessions(
    stream: ScoredSplit, held_out: Collection[str], lookback_s: float = DEFAULT_LOOKBACK_S
) -> list[dict[str, Any]]:
    """
    Build attack and stable sessions from a stream (decision D5).

    Per device, time-sorted, each contiguous run of held-out flows (consecutive gaps
    <= lookback_s) is one attack session with window [first held-out time - lookback_s,
    last held-out time] and change_point = first held-out time. Stable sessions are
    non-overlapping lookback_s windows per device that contain no held-out flow.

    Returns:
        Dicts with `session_type`, `device`, `index` (stream row positions, time-sorted)
        and `change_point` (None for stable sessions).
    """
    is_held_out = np.isin(stream.y_true, list(held_out))
    sessions: list[dict[str, Any]] = []
    for device in pd.unique(stream.device):
        rows = np.flatnonzero(stream.device == device)
        rows = rows[np.argsort(stream.timestamp[rows], kind='stable')]
        times = stream.timestamp[rows]
        held = is_held_out[rows]
        held_times = times[held]

        if held_times.size:
            breaks = np.flatnonzero(np.diff(held_times) > lookback_s) + 1
            for burst in np.split(held_times, breaks):
                start, end = float(burst[0]), float(burst[-1])
                in_window = (times >= start - lookback_s) & (times <= end)
                sessions.append({
                    'session_type': 'attack',
                    'device': device,
                    'index': rows[in_window],
                    'change_point': start,
                })

        window_start = float(times[0])
        while window_start <= times[-1]:
            in_window = (times >= window_start) & (times < window_start + lookback_s)
            if in_window.any() and not held[in_window].any():
                sessions.append({
                    'session_type': 'stable',
                    'device': device,
                    'index': rows[in_window],
                    'change_point': None,
                })
            window_start += lookback_s
    return sessions


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _delay_metrics(
    stream: ScoredSplit, flags: np.ndarray, sessions: Sequence[dict[str, Any]]
) -> dict[str, float | int | None]:
    delays_s: list[float] = []
    delays_flows: list[float] = []
    n_attack = n_detected = stable_alerts = stable_flows = 0
    for session in sessions:
        index = session['index']
        result = evaluate_detection_delay(
            stream.timestamp[index], flags[index], session['session_type'], session['change_point']
        )
        if session['session_type'] == 'attack':
            n_attack += 1
            if result['detected']:
                n_detected += 1
                delays_s.append(result['delay_seconds'])
                delays_flows.append(result['delay_flows'])
        else:
            stable_alerts += result['false_alerts']
            stable_flows += result['n_flows']
    return {
        'n_attack_sessions': n_attack,
        'n_detected': n_detected,
        'median_delay_s': _median(delays_s),
        'median_delay_flows': _median(delays_flows),
        'stable_fa_rate': stable_alerts / stable_flows if stable_flows else None,
    }


def evaluate_seed(
    val: ScoredSplit,
    stream: ScoredSplit,
    held_out: Collection[str],
    class_names: Sequence[str],
    fa_targets: Sequence[float] = DEFAULT_FA_TARGETS,
    lookback_s: float = DEFAULT_LOOKBACK_S,
) -> list[dict[str, Any]]:
    """
    Evaluate every rule at every false-alert target for one seed.

    Args:
        val: Known-class validation split used to select thresholds.
        stream: Stream split containing known and held-out flows.
        held_out: Class names that count as unknown.
        class_names: Class name of each `probas` column, in column order. Callers decode
            `model.classes_` through the label encoder; these are the names `y_true` uses.
        fa_targets: Matched false-alert targets on val.
        lookback_s: Session lookback window in seconds.

    Returns:
        One row per rule in NOVELTY_RULES and target in fa_targets, with the threshold,
        stream known-class false-alert rate, open-world, closed-world and session delay
        metrics. Novelty scores for AUROC/AUPRC are 1 - max_softmax (confidence),
        1 - max p-value (conformal) and 1 - max(max_softmax, max p-value) (combined).
    """
    names = np.asarray(class_names)
    if names.shape[0] != stream.probas.shape[1]:
        raise ValueError(f'class_names has {names.shape[0]} entries for {stream.probas.shape[1]} probability columns')
    unknown = np.isin(stream.y_true, list(held_out))
    known = ~unknown
    sessions = build_sessions(stream, held_out, lookback_s)
    closed = evaluate_closed_world(stream.y_true[known], names[np.argmax(stream.probas[known], axis=1)])

    rows: list[dict[str, Any]] = []
    for rule in NOVELTY_RULES:
        scores = novelty_scores(rule, stream)
        for target in fa_targets:
            chosen = select_thresholds(rule, val.probas, val.p_values, target)
            flags = apply_novelty_rule(rule, stream.probas, stream.p_values, chosen['tau'], chosen['alpha'])
            open_world = evaluate_open_world_novelty(unknown, flags, scores)
            rows.append({
                'rule': rule,
                'target_fa': target,
                'tau': chosen['tau'],
                'alpha': chosen['alpha'],
                'val_fa': chosen['achieved_fa'],
                'stream_known_fa': float(np.mean(flags[known])),
                'unknown_precision': open_world['precision'],
                'unknown_recall': open_world['recall'],
                'unknown_f1': open_world['f1'],
                'auroc': open_world['auroc'],
                'auprc': open_world['auprc'],
                'alert_burden': open_world['alert_burden'],
                'closed_macro_f1': closed['macro_f1'],
                **_delay_metrics(stream, flags, sessions),
            })
    return rows


def build_scores_frame(stream: ScoredSplit, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Per-flow stream rows with a `flag_<rule>_<target>` column for each row of `evaluate_seed`."""
    frame = pd.DataFrame({
        'device': stream.device,
        'timestamp': stream.timestamp,
        'y_true': stream.y_true,
        'max_softmax': np.max(stream.probas, axis=1),
        'max_p': _max_p(stream),
    })
    for row in rows:
        flags = apply_novelty_rule(row['rule'], stream.probas, stream.p_values, row['tau'], row['alpha'])
        frame[f'flag_{row["rule"]}_{row["target_fa"]}'] = flags
    return frame


def summarize(per_seed: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    """One row per rule x target with `<metric>_mean`, `<metric>_std` across seeds and `n_seeds`."""
    table = pd.DataFrame([row for rows in per_seed.values() for row in rows])
    metrics = [c for c in table.columns if c not in _KEY_COLUMNS]
    table[metrics] = table[metrics].apply(pd.to_numeric)
    grouped = table.groupby(list(_KEY_COLUMNS), sort=False)[metrics].agg(['mean', 'std'])
    grouped.columns = [f'{metric}_{stat}' for metric, stat in grouped.columns]
    grouped['n_seeds'] = table.groupby(list(_KEY_COLUMNS), sort=False).size()
    return grouped.reset_index()


def write_results(
    out_dir: Path, config: dict, per_seed: dict[int, list[dict[str, Any]]], scores: dict[int, pd.DataFrame]
) -> None:
    """
    Write the results layout under out_dir.

    Files: `config.json`, `seed_<k>/metrics.json`, `seed_<k>/scores.parquet` and `summary.csv`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    for seed, rows in per_seed.items():
        seed_dir = out_dir / f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        (seed_dir / 'metrics.json').write_text(json.dumps(rows, indent=2))
        scores[seed].to_parquet(seed_dir / 'scores.parquet', index=False)
    summarize(per_seed).to_csv(out_dir / 'summary.csv', index=False)


def make_run_id() -> str:
    """Return `<UTC YYYYmmddTHHMMSSZ>-<git short sha or 'nogit'>`."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    try:
        sha = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = ''
    return f'{stamp}-{sha or "nogit"}'
