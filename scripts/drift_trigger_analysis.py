from __future__ import annotations

import argparse
import json
import logging
import math
import re

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import DefaultDict, Iterable, TextIO

LOGGER = logging.getLogger(__name__)

EXPECTED_CE_TYPES: tuple[str, ...] = (
    'ice',
    'approx_cce',
    'cce',
    'approx_tce',
    'none',
)

FILENAME_RE = re.compile(
    r"""
    ^
    (?P<classifier>.+?)_
    (?P<ce_type>ice|approx_cce|cce|approx_tce|none)_
    (?P<model_type>mc|binary)_
    (?P<seed>\d+)_
    (?P<run>\d+)_run
    \.log$
    """,
    re.VERBOSE,
)

PREDICTION_RE = re.compile(
    r"""
    \[Index\s+(?P<index>\d+)\]\s+
    Predicted=(?P<predicted>\d+),\s+
    Actual=(?P<actual>\d+)
    """,
    re.VERBOSE,
)

DRIFT_RE = re.compile(r'Drift detected in the chunk\. Retraining model and recalibrating CE\.\.\.')

NO_DRIFT_RE = re.compile(
    r"""
    \[AdaptiveChunking\]\s+
    Chunk\s+size\s+changed\s+from\s+
    (?P<old_size>\d+)\s+to\s+(?P<new_size>\d+)
    \s+\(drift\s+EMA:\s+(?P<ema>[0-9.]+)\)
    """,
    re.VERBOSE,
)


@dataclass(slots=True, frozen=True)
class RunIdentity:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    seed: int
    run: int
    filename: str


@dataclass(slots=True)
class ChunkObservation:
    identity: RunIdentity
    chunk_id: int
    num_samples: int
    num_attacks: int
    num_benign: int
    retrain_triggered: bool
    ended_by: str

    @property
    def attack_present(self) -> bool:
        return self.num_attacks > 0

    @property
    def attack_ratio(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.num_attacks / self.num_samples


@dataclass(slots=True)
class CurrentChunk:
    num_samples: int = 0
    num_attacks: int = 0
    num_benign: int = 0

    def add_actual(self, actual: int) -> None:
        self.num_samples += 1
        if actual == 1:
            self.num_attacks += 1
        else:
            self.num_benign += 1

    def reset(self) -> None:
        self.num_samples = 0
        self.num_attacks = 0
        self.num_benign = 0


@dataclass(slots=True, frozen=True)
class GroupKey:
    dataset: str
    ce_type: str


@dataclass(slots=True)
class ContingencyStats:
    group_name: str
    total_chunks: int
    attack_chunks: int
    non_attack_chunks: int
    retrain_chunks: int
    no_retrain_chunks: int

    attack_and_retrain: int
    attack_and_no_retrain: int
    no_attack_and_retrain: int
    no_attack_and_no_retrain: int

    p_retrain_given_attack: float
    p_retrain_given_no_attack: float
    risk_difference: float
    risk_ratio: float | None
    odds_ratio: float | None
    fisher_two_sided_p: float | None
    phi: float | None


@dataclass(slots=True)
class AnalysisResult:
    observations: list[ChunkObservation] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    parsed_files: list[str] = field(default_factory=list)


def parse_identity(file_path: Path, log_dir: Path) -> RunIdentity:
    match = FILENAME_RE.match(file_path.name)
    if match is None:
        raise ValueError(f'Unrecognized log filename format: {file_path.name}')

    return RunIdentity(
        dataset=file_path.parent.name,
        classifier=match.group('classifier'),
        ce_type=match.group('ce_type'),
        model_type=match.group('model_type'),
        seed=int(match.group('seed')),
        run=int(match.group('run')),
        filename=str(file_path.relative_to(log_dir)),
    )


def iter_log_files(log_dir: Path) -> Iterable[Path]:
    for file_path in sorted(log_dir.rglob('*.log')):
        if not file_path.is_file():
            continue

        match = FILENAME_RE.match(file_path.name)
        if match is None:
            LOGGER.debug('Skipping non-run log: %s', file_path)
            continue

        if match.group('ce_type') == 'none':
            LOGGER.debug('Skipping CE=none log: %s', file_path)
            continue

        yield file_path


def close_chunk(
    *,
    observations: list[ChunkObservation],
    identity: RunIdentity,
    current_chunk: CurrentChunk,
    chunk_id: int,
    retrain_triggered: bool,
    ended_by: str,
) -> int:
    if current_chunk.num_samples == 0:
        return chunk_id

    observations.append(
        ChunkObservation(
            identity=identity,
            chunk_id=chunk_id,
            num_samples=current_chunk.num_samples,
            num_attacks=current_chunk.num_attacks,
            num_benign=current_chunk.num_benign,
            retrain_triggered=retrain_triggered,
            ended_by=ended_by,
        )
    )
    current_chunk.reset()
    return chunk_id + 1


def parse_log_file(file_path: Path, log_dir: Path) -> list[ChunkObservation]:
    identity = parse_identity(file_path, log_dir)

    observations: list[ChunkObservation] = []
    current_chunk = CurrentChunk()
    chunk_id = 0

    with file_path.open('r', encoding='utf-8', errors='replace') as handle:
        for raw_line in handle:
            line = raw_line.strip()

            prediction_match = PREDICTION_RE.search(line)
            if prediction_match is not None:
                actual = int(prediction_match.group('actual'))
                current_chunk.add_actual(actual)
                continue

            if DRIFT_RE.search(line) is not None:
                chunk_id = close_chunk(
                    observations=observations,
                    identity=identity,
                    current_chunk=current_chunk,
                    chunk_id=chunk_id,
                    retrain_triggered=True,
                    ended_by='drift_detected',
                )
                continue

            if NO_DRIFT_RE.search(line) is not None:
                chunk_id = close_chunk(
                    observations=observations,
                    identity=identity,
                    current_chunk=current_chunk,
                    chunk_id=chunk_id,
                    retrain_triggered=False,
                    ended_by='adaptive_chunking_no_drift',
                )
                continue

    # NO force closing trailing preds w/o a chunk outcome
    # Those rows cannot be assigned to drift/no-drift w/ conf
    if current_chunk.num_samples > 0:
        LOGGER.debug(
            'Ignoring trailing incomplete chunk in %s with %d samples',
            file_path,
            current_chunk.num_samples,
        )

    return observations


def analyze_logs(log_dir: Path) -> AnalysisResult:
    result = AnalysisResult()

    if not log_dir.exists():
        raise FileNotFoundError(f'Log directory does not exist: {log_dir}')

    for file_path in iter_log_files(log_dir):
        try:
            observations = parse_log_file(file_path=file_path, log_dir=log_dir)
        except Exception as exc:
            LOGGER.exception('Failed to parse %s', file_path)
            result.skipped_files.append(f'{file_path}: {exc}')
            continue

        if not observations:
            LOGGER.warning('No chunk observations parsed from %s', file_path)

        result.parsed_files.append(str(file_path.relative_to(log_dir)))
        result.observations.extend(observations)

    return result


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def odds_ratio_haldane_anscombe(
    a: int,
    b: int,
    c: int,
    d: int,
) -> float:
    # h-a corr avoids infinite || w/ sparse cells
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def risk_ratio_haldane_anscombe(
    a: int,
    b: int,
    c: int,
    d: int,
) -> float:
    exposed_total = a + b
    unexposed_total = c + d

    exposed_risk = (a + 0.5) / (exposed_total + 1.0)
    unexposed_risk = (c + 0.5) / (unexposed_total + 1.0)

    return exposed_risk / unexposed_risk


def hypergeom_probability(
    *,
    a: int,
    row1: int,
    row2: int,
    col1: int,
    total: int,
) -> float:
    return math.comb(row1, a) * math.comb(row2, col1 - a) / math.comb(total, col1)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float | None:
    """
    Pure-Python Fisher exact test for a 2x2 table.

    Table layout:
        attack chunk      + retrain: a
        attack chunk      + no retrain: b
        no attack chunk   + retrain: c
        no attack chunk   + no retrain: d
    """
    total = a + b + c + d
    if total == 0:
        return None

    row1 = a + b
    row2 = c + d
    col1 = a + c

    min_a = max(0, col1 - row2)
    max_a = min(row1, col1)

    observed_p = hypergeom_probability(
        a=a,
        row1=row1,
        row2=row2,
        col1=col1,
        total=total,
    )

    p_value = 0.0
    for possible_a in range(min_a, max_a + 1):
        prob = hypergeom_probability(
            a=possible_a,
            row1=row1,
            row2=row2,
            col1=col1,
            total=total,
        )
        if prob <= observed_p + 1e-12:
            p_value += prob

    return min(p_value, 1.0)


def phi_coefficient(a: int, b: int, c: int, d: int) -> float | None:
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        return None
    return ((a * d) - (b * c)) / denominator


def build_contingency_stats(
    group_name: str,
    observations: list[ChunkObservation],
) -> ContingencyStats:
    a = sum(1 for obs in observations if obs.attack_present and obs.retrain_triggered)
    b = sum(1 for obs in observations if obs.attack_present and not obs.retrain_triggered)
    c = sum(1 for obs in observations if not obs.attack_present and obs.retrain_triggered)
    d = sum(1 for obs in observations if not obs.attack_present and not obs.retrain_triggered)

    attack_chunks = a + b
    non_attack_chunks = c + d
    retrain_chunks = a + c
    no_retrain_chunks = b + d
    total_chunks = attack_chunks + non_attack_chunks

    p_attack = safe_divide(a, attack_chunks)
    p_no_attack = safe_divide(c, non_attack_chunks)

    return ContingencyStats(
        group_name=group_name,
        total_chunks=total_chunks,
        attack_chunks=attack_chunks,
        non_attack_chunks=non_attack_chunks,
        retrain_chunks=retrain_chunks,
        no_retrain_chunks=no_retrain_chunks,
        attack_and_retrain=a,
        attack_and_no_retrain=b,
        no_attack_and_retrain=c,
        no_attack_and_no_retrain=d,
        p_retrain_given_attack=p_attack,
        p_retrain_given_no_attack=p_no_attack,
        risk_difference=p_attack - p_no_attack,
        risk_ratio=risk_ratio_haldane_anscombe(a, b, c, d) if total_chunks > 0 else None,
        odds_ratio=odds_ratio_haldane_anscombe(a, b, c, d) if total_chunks > 0 else None,
        fisher_two_sided_p=fisher_exact_two_sided(a, b, c, d),
        phi=phi_coefficient(a, b, c, d),
    )


def group_observations(
    observations: list[ChunkObservation],
) -> tuple[
    dict[str, list[ChunkObservation]],
    dict[str, list[ChunkObservation]],
    dict[str, list[ChunkObservation]],
]:
    by_dataset: DefaultDict[str, list[ChunkObservation]] = defaultdict(list)
    by_ce_type: DefaultDict[str, list[ChunkObservation]] = defaultdict(list)
    by_dataset_ce: DefaultDict[str, list[ChunkObservation]] = defaultdict(list)

    for obs in observations:
        dataset = obs.identity.dataset
        ce_type = obs.identity.ce_type

        by_dataset[dataset].append(obs)
        by_ce_type[ce_type].append(obs)
        by_dataset_ce[f'{dataset} | {ce_type}'].append(obs)

    return dict(by_dataset), dict(by_ce_type), dict(by_dataset_ce)


def format_optional_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def write_stats_section(
    handle: TextIO,
    title: str,
    stats: list[ContingencyStats],
) -> None:
    handle.write(f'\n=== {title} ===\n\n')

    for item in sorted(stats, key=lambda value: value.group_name):
        handle.write(f'{item.group_name}\n')
        handle.write(f'  total chunks = {item.total_chunks}\n')
        handle.write(f'  attack chunks = {item.attack_chunks}\n')
        handle.write(f'  non-attack chunks = {item.non_attack_chunks}\n')
        handle.write(f'  retrain chunks = {item.retrain_chunks}\n')
        handle.write(f'  no-retrain chunks = {item.no_retrain_chunks}\n')
        handle.write('  contingency table:\n')
        handle.write('                  retrain    no_retrain\n')
        handle.write(f'    attack       {item.attack_and_retrain:8d}  {item.attack_and_no_retrain:10d}\n')
        handle.write(f'    no_attack    {item.no_attack_and_retrain:8d}  {item.no_attack_and_no_retrain:10d}\n')
        handle.write(f'  P(retrain | attack chunk) = {item.p_retrain_given_attack:.6f}\n')
        handle.write(f'  P(retrain | no-attack chunk) = {item.p_retrain_given_no_attack:.6f}\n')
        handle.write(f'  risk difference = {item.risk_difference:.6f}\n')
        handle.write(f'  risk ratio = {format_optional_float(item.risk_ratio)}\n')
        handle.write(f'  odds ratio = {format_optional_float(item.odds_ratio)}\n')
        handle.write(f'  Fisher exact two-sided p = {format_optional_float(item.fisher_two_sided_p)}\n')
        handle.write(f'  phi = {format_optional_float(item.phi)}\n\n')


def build_all_stats(observations: list[ChunkObservation]) -> dict[str, list[ContingencyStats]]:
    by_dataset, by_ce_type, by_dataset_ce = group_observations(observations)

    return {
        'overall': [build_contingency_stats('OVERALL', observations)],
        'by_dataset': [build_contingency_stats(name, group) for name, group in by_dataset.items()],
        'by_ce_type': [build_contingency_stats(name, group) for name, group in by_ce_type.items()],
        'by_dataset_ce_type': [build_contingency_stats(name, group) for name, group in by_dataset_ce.items()],
    }


def write_output(
    *,
    output_file: Path,
    log_dir: Path,
    result: AnalysisResult,
    grouped_stats: dict[str, list[ContingencyStats]],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w', encoding='utf-8') as handle:
        handle.write('=== Drift Trigger Attack Association Analysis ===\n\n')
        handle.write(f'log_dir = {log_dir}\n')
        handle.write(f'parsed_files = {len(result.parsed_files)}\n')
        handle.write(f'skipped_files = {len(result.skipped_files)}\n')
        handle.write(f'chunk_observations = {len(result.observations)}\n')
        handle.write('\nInterpretation:\n')
        handle.write('  This analysis treats each completed chunk as one observation.\n')
        handle.write('  attack chunk = at least one prediction line in the chunk had Actual=1.\n')
        handle.write('  retrain chunk = the chunk ended with the drift/retraining log line.\n')
        handle.write('  no-retrain chunk = the chunk ended with an AdaptiveChunking chunk-size-change line.\n')
        handle.write(
            '  If attacks directly trigger retraining, P(retrain | attack chunk) '
            'should be much larger than P(retrain | no-attack chunk), with a '
            'large positive risk difference, odds ratio > 1, and small Fisher p-value.\n'
        )

        if result.skipped_files:
            handle.write('\nSkipped files:\n')
            for item in result.skipped_files:
                handle.write(f'  {item}\n')

        write_stats_section(handle, 'Overall', grouped_stats['overall'])
        write_stats_section(handle, 'By Dataset', grouped_stats['by_dataset'])
        write_stats_section(handle, 'By CE Type', grouped_stats['by_ce_type'])
        write_stats_section(
            handle,
            'By Dataset and CE Type',
            grouped_stats['by_dataset_ce_type'],
        )


def write_json_output(
    *,
    output_file: Path,
    result: AnalysisResult,
    grouped_stats: dict[str, list[ContingencyStats]],
) -> None:
    json_file = output_file.with_suffix('.json')
    payload = {
        'parsed_files': result.parsed_files,
        'skipped_files': result.skipped_files,
        'num_observations': len(result.observations),
        'grouped_stats': {key: [asdict(item) for item in values] for key, values in grouped_stats.items()},
        'observations': [
            {
                **asdict(obs),
                'identity': asdict(obs.identity),
                'attack_present': obs.attack_present,
                'attack_ratio': obs.attack_ratio,
            }
            for obs in result.observations
        ],
    }

    with json_file.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=('Analyze whether attack-containing chunks are associated with drift-triggered retraining.')
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('../seed_runs/ac'),
        help='Directory containing dataset subdirectories of run logs.',
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('logging/seed_runs/drift_trigger_analysis.log'),
        help='Text output path.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging.',
    )
    return parser


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(levelname)s: %(message)s',
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(debug=args.debug)

    result = analyze_logs(log_dir=args.log_dir)
    grouped_stats = build_all_stats(result.observations)

    write_output(
        output_file=args.output_file,
        log_dir=args.log_dir,
        result=result,
        grouped_stats=grouped_stats,
    )
    write_json_output(
        output_file=args.output_file,
        result=result,
        grouped_stats=grouped_stats,
    )

    LOGGER.info('Wrote analysis to %s', args.output_file)
    LOGGER.info('Wrote JSON to %s', args.output_file.with_suffix('.json'))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
