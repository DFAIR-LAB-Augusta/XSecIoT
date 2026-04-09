from __future__ import annotations

import argparse
import json
import logging
import re

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, DefaultDict, Iterable, TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


EXPECTED_SEEDS: tuple[int, ...] = (17, 42, 67, 92, 117)
EXPECTED_RUNS: tuple[int, ...] = (0, 1, 2, 3, 4)
EXPECTED_CE_TYPES: tuple[str, ...] = (
    'ice',
    'approx_cce',
    'cce',
    'approx_tce',
    'none',
)


class ScraperConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_dir: Path = Field(default=Path('./logging/ac'))
    output_file: Path | None = None
    json_output_file: Path | None = None
    stat_marker: str = '[==OVERALL SIM STATS==]'
    exclude_dir_prefix: str = 'old'
    log_suffix: str = '.log'

    chunk_stats: frozenset[str] = frozenset({
        'Average Chunk Size',
        'Median Chunk Size',
        'Standard Deviation of Chunk Sizes',
    })

    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS
    expected_runs: tuple[int, ...] = EXPECTED_RUNS
    expected_ce_types: tuple[str, ...] = EXPECTED_CE_TYPES

    @field_validator('log_dir', mode='before')
    @classmethod
    def _coerce_log_dir(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator('output_file', 'json_output_file', mode='before')
    @classmethod
    def _coerce_optional_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value)

    def resolved_output_file(self) -> Path:
        if self.output_file is not None:
            return self.output_file
        return self.log_dir / 'overall_stats.log'

    def resolved_json_output_file(self) -> Path:
        if self.json_output_file is not None:
            return self.json_output_file
        return self.log_dir / 'overall_stats.json'


@dataclass(slots=True, frozen=True)
class RunIdentity:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    seed: int
    run: int
    filename: str


@dataclass(slots=True, frozen=True)
class StatEntry:
    identity: RunIdentity
    subfolder: str
    stat_key: str
    raw_value: str
    numeric_value: float | None


@dataclass(slots=True)
class FileStats:
    identity: RunIdentity
    subfolder: str
    stat_lines: list[str] = field(default_factory=list)
    entries: list[StatEntry] = field(default_factory=list)

    def add_entry(self, entry: StatEntry) -> None:
        self.entries.append(entry)
        self.stat_lines.append(f'{entry.stat_key}: {entry.raw_value}')


@dataclass(slots=True, frozen=True)
class GroupKey:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    stat_key: str


@dataclass(slots=True)
class GroupSummary:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    stat_key: str
    n: int
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    seeds_present: list[int]
    runs_present: list[int]
    files: list[str]


@dataclass(slots=True)
class CoverageSummary:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    expected_total: int
    found_total: int
    missing_pairs: list[str]


@dataclass(slots=True)
class ParseResult:
    by_subfolder: DefaultDict[str, dict[str, FileStats]] = field(default_factory=lambda: defaultdict(dict))
    by_stat: DefaultDict[str, list[StatEntry]] = field(default_factory=lambda: defaultdict(list))
    all_files: list[FileStats] = field(default_factory=list)

    def add_file_stats(self, file_stats: FileStats) -> None:
        self.by_subfolder[file_stats.subfolder][file_stats.identity.filename] = file_stats
        self.all_files.append(file_stats)

        for entry in file_stats.entries:
            if entry.numeric_value is not None:
                self.by_stat[entry.stat_key].append(entry)


class OverallStatsScraper:
    CE_SUMMARY_MARKER = '=== CE Model Calibration Summary ==='
    CLASSIFIER_SUMMARY_MARKER = '=== Classifier Model Performance Summary ==='

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

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.key_value_pattern = re.compile(rf'{re.escape(config.stat_marker)}\s+(.*?):\s+(.*)')

    def run(self) -> tuple[Path, Path]:
        logger.info('Starting scrape from %s', self.config.log_dir)
        result = self.parse_logs()
        grouped = self.build_group_summaries(result)
        coverage = self.build_coverage_summaries(result)

        log_output_path = self.write_text_output(result, grouped, coverage)
        json_output_path = self.write_json_output(result, grouped, coverage)

        logger.info('Wrote text summary to %s', log_output_path)
        logger.info('Wrote JSON summary to %s', json_output_path)
        return log_output_path, json_output_path

    def parse_logs(self) -> ParseResult:
        result = ParseResult()

        for file_path in self.iter_log_files():
            try:
                file_stats = self.parse_log_file(file_path)
            except Exception:
                logger.exception('Failed to parse log file: %s', file_path)
                continue

            if file_stats.entries:
                result.add_file_stats(file_stats)

        return result

    def iter_log_files(self) -> Iterable[Path]:
        if not self.config.log_dir.exists():
            logger.warning('Log directory does not exist: %s', self.config.log_dir)
            return []

        for file_path in self.config.log_dir.rglob(f'*{self.config.log_suffix}'):
            relative_parts = file_path.relative_to(self.config.log_dir).parts
            if any(part.startswith(self.config.exclude_dir_prefix) for part in relative_parts[:-1]):
                logger.debug('Skipping excluded file: %s', file_path)
                continue
            yield file_path

    def parse_log_file(self, file_path: Path) -> FileStats:
        identity = self.parse_identity(file_path)
        subfolder = self._relative_subfolder(file_path.parent)
        file_stats = FileStats(identity=identity, subfolder=subfolder)

        model_section: str | None = None

        with file_path.open('r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()

                if self.CE_SUMMARY_MARKER in line:
                    model_section = 'CE Model'
                    continue

                if self.CLASSIFIER_SUMMARY_MARKER in line:
                    model_section = 'Classifier Model'
                    continue

                if self.config.stat_marker not in line:
                    continue

                entry = self.parse_stat_line(
                    line=line,
                    identity=identity,
                    subfolder=subfolder,
                    model_section=model_section,
                )
                if entry is not None:
                    file_stats.add_entry(entry)

        return file_stats

    def parse_identity(self, file_path: Path) -> RunIdentity:
        dataset = file_path.parent.name
        match = self.FILENAME_RE.match(file_path.name)

        if match is None:
            raise ValueError(f'Unrecognized log filename format: {file_path.name}')

        return RunIdentity(
            dataset=dataset,
            classifier=match.group('classifier'),
            ce_type=match.group('ce_type'),
            model_type=match.group('model_type'),
            seed=int(match.group('seed')),
            run=int(match.group('run')),
            filename=file_path.name,
        )

    def parse_stat_line(
        self,
        *,
        line: str,
        identity: RunIdentity,
        subfolder: str,
        model_section: str | None,
    ) -> StatEntry | None:
        match = self.key_value_pattern.search(line)
        if not match:
            logger.debug('Could not parse stat line: %s', line)
            return None

        stat_key = match.group(1).strip()
        stat_value = match.group(2).strip()
        full_key = self.build_full_key(stat_key, model_section)
        numeric_value = self.parse_numeric_value(stat_value)

        return StatEntry(
            identity=identity,
            subfolder=subfolder,
            stat_key=full_key,
            raw_value=stat_value,
            numeric_value=numeric_value,
        )

    def build_full_key(self, stat_key: str, model_section: str | None) -> str:
        if stat_key in self.config.chunk_stats:
            return stat_key

        prefix = f'[{model_section}]' if model_section else ''
        return f'{prefix} {stat_key}'.strip()

    @staticmethod
    def parse_numeric_value(raw_value: str) -> float | None:
        cleaned = raw_value.strip().replace(',', '')
        cleaned = cleaned.rstrip('%s')

        try:
            return float(cleaned)
        except ValueError:
            return None

    def build_group_summaries(self, result: ParseResult) -> list[GroupSummary]:
        grouped: DefaultDict[GroupKey, list[StatEntry]] = defaultdict(list)

        for file_stats in result.all_files:
            for entry in file_stats.entries:
                if entry.numeric_value is None:
                    continue

                key = GroupKey(
                    dataset=entry.identity.dataset,
                    classifier=entry.identity.classifier,
                    ce_type=entry.identity.ce_type,
                    model_type=entry.identity.model_type,
                    stat_key=entry.stat_key,
                )
                grouped[key].append(entry)

        summaries: list[GroupSummary] = []

        for key in sorted(
            grouped,
            key=lambda item: (
                item.dataset,
                item.classifier,
                item.model_type,
                item.ce_type,
                item.stat_key,
            ),
        ):
            entries = grouped[key]
            values = [entry.numeric_value for entry in entries if entry.numeric_value is not None]
            if not values:
                continue

            seeds_present = sorted({entry.identity.seed for entry in entries})
            runs_present = sorted({entry.identity.run for entry in entries})
            files = sorted({entry.identity.filename for entry in entries})

            summaries.append(
                GroupSummary(
                    dataset=key.dataset,
                    classifier=key.classifier,
                    ce_type=key.ce_type,
                    model_type=key.model_type,
                    stat_key=key.stat_key,
                    n=len(values),
                    mean_value=mean(values),
                    std_value=stdev(values) if len(values) > 1 else 0.0,
                    min_value=min(values),
                    max_value=max(values),
                    seeds_present=seeds_present,
                    runs_present=runs_present,
                    files=files,
                )
            )

        return summaries

    def build_coverage_summaries(self, result: ParseResult) -> list[CoverageSummary]:
        seen: DefaultDict[tuple[str, str, str, str], set[tuple[int, int]]] = defaultdict(set)

        for file_stats in result.all_files:
            identity = file_stats.identity
            key = (
                identity.dataset,
                identity.classifier,
                identity.ce_type,
                identity.model_type,
            )
            seen[key].add((identity.seed, identity.run))

        coverage: list[CoverageSummary] = []
        expected_pairs = {(seed, run) for seed in self.config.expected_seeds for run in self.config.expected_runs}
        expected_total = len(expected_pairs)

        for key in sorted(seen):
            found_pairs = seen[key]
            missing_pairs = sorted(expected_pairs - found_pairs)
            coverage.append(
                CoverageSummary(
                    dataset=key[0],
                    classifier=key[1],
                    ce_type=key[2],
                    model_type=key[3],
                    expected_total=expected_total,
                    found_total=len(found_pairs),
                    missing_pairs=[f'seed={seed}, run={run}' for seed, run in missing_pairs],
                )
            )

        return coverage

    def write_text_output(
        self,
        result: ParseResult,
        grouped: list[GroupSummary],
        coverage: list[CoverageSummary],
    ) -> Path:
        output_file = self.config.resolved_output_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open('w', encoding='utf-8') as out:
            self.write_parsed_stats(out, result)
            self.write_grouped_summary(out, grouped)
            self.write_coverage_summary(out, coverage)

        return output_file

    def write_parsed_stats(self, handle: TextIO, result: ParseResult) -> None:
        handle.write('=== Parsed [==OVERALL SIM STATS==] by Subfolder and File ===\n\n')

        for subfolder in sorted(result.by_subfolder):
            handle.write(f'Subfolder: {subfolder}\n')

            for filename in sorted(result.by_subfolder[subfolder]):
                file_stats = result.by_subfolder[subfolder][filename]
                meta = file_stats.identity
                handle.write(f'  File: {filename}\n')
                handle.write(
                    '    Meta: '
                    f'dataset={meta.dataset}, classifier={meta.classifier}, '
                    f'ce_type={meta.ce_type}, model_type={meta.model_type}, '
                    f'seed={meta.seed}, run={meta.run}\n'
                )

                for stat_line in file_stats.stat_lines:
                    handle.write(f'    {stat_line}\n')

            handle.write('\n')

    def write_grouped_summary(
        self,
        handle: TextIO,
        grouped: list[GroupSummary],
    ) -> None:
        handle.write('=== Grouped Mean ± Std by Experiment Configuration ===\n\n')

        current_group: tuple[str, str, str, str] | None = None

        for summary in grouped:
            group_key = (
                summary.dataset,
                summary.classifier,
                summary.ce_type,
                summary.model_type,
            )

            if current_group != group_key:
                current_group = group_key
                handle.write(
                    f'Dataset={summary.dataset} | '
                    f'Classifier={summary.classifier} | '
                    f'CE={summary.ce_type} | '
                    f'Mode={summary.model_type}\n'
                )

            handle.write(f'  Stat: {summary.stat_key}\n')
            handle.write(f'    n = {summary.n}\n')
            handle.write(f'    mean ± std = {summary.mean_value:.6f} ± {summary.std_value:.6f}\n')
            handle.write(f'    min = {summary.min_value:.6f}\n')
            handle.write(f'    max = {summary.max_value:.6f}\n')
            handle.write(f'    seeds present = {summary.seeds_present}\n')
            handle.write(f'    runs present = {summary.runs_present}\n')

        handle.write('\n')

    def write_coverage_summary(
        self,
        handle: TextIO,
        coverage: list[CoverageSummary],
    ) -> None:
        handle.write('=== Coverage Summary ===\n\n')

        for item in coverage:
            handle.write(
                f'Dataset={item.dataset} | Classifier={item.classifier} | CE={item.ce_type} | Mode={item.model_type}\n'
            )
            handle.write(f'  found = {item.found_total}/{item.expected_total}\n')

            if item.missing_pairs:
                handle.write('  missing:\n')
                for pair in item.missing_pairs:
                    handle.write(f'    {pair}\n')
            else:
                handle.write('  missing: none\n')

        handle.write('\n')

    def write_json_output(
        self,
        result: ParseResult,
        grouped: list[GroupSummary],
        coverage: list[CoverageSummary],
    ) -> Path:
        output_file = self.config.resolved_json_output_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        payload = self.build_json_payload(result, grouped, coverage)

        with output_file.open('w', encoding='utf-8') as out:
            json.dump(payload, out, indent=2, sort_keys=True)

        return output_file

    def build_json_payload(
        self,
        result: ParseResult,
        grouped: list[GroupSummary],
        coverage: list[CoverageSummary],
    ) -> dict[str, Any]:
        return {
            'config': {
                'log_dir': str(self.config.log_dir),
                'output_file': str(self.config.resolved_output_file()),
                'json_output_file': str(self.config.resolved_json_output_file()),
                'stat_marker': self.config.stat_marker,
                'exclude_dir_prefix': self.config.exclude_dir_prefix,
                'log_suffix': self.config.log_suffix,
                'expected_seeds': list(self.config.expected_seeds),
                'expected_runs': list(self.config.expected_runs),
                'expected_ce_types': list(self.config.expected_ce_types),
            },
            'parsed': self._build_parsed_json(result),
            'grouped_summary': [asdict(item) for item in grouped],
            'coverage_summary': [asdict(item) for item in coverage],
        }

    def _build_parsed_json(self, result: ParseResult) -> dict[str, dict[str, Any]]:
        parsed: dict[str, dict[str, Any]] = {}

        for subfolder in sorted(result.by_subfolder):
            parsed[subfolder] = {}
            for filename in sorted(result.by_subfolder[subfolder]):
                file_stats = result.by_subfolder[subfolder][filename]
                parsed[subfolder][filename] = {
                    'identity': asdict(file_stats.identity),
                    'stat_lines': file_stats.stat_lines,
                    'entries': [asdict(entry) for entry in file_stats.entries],
                }

        return parsed

    def _relative_subfolder(self, directory: Path) -> str:
        relative = directory.relative_to(self.config.log_dir)
        return relative.as_posix() if relative.parts else '.'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Scrape FIRCE overall stats from log files.')
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('./logging/ac'),
        help='Directory containing FIRCE log files.',
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=None,
        help='Text output file path. Defaults to <log-dir>/overall_stats.log',
    )
    parser.add_argument(
        '--json-output-file',
        type=Path,
        default=None,
        help='JSON output file path. Defaults to <log-dir>/overall_stats.json',
    )
    parser.add_argument(
        '--exclude-dir-prefix',
        type=str,
        default='old',
        help='Skip directories whose names start with this prefix.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging.',
    )
    return parser


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(debug=args.debug)

    config = ScraperConfig(
        log_dir=args.log_dir,
        output_file=args.output_file,
        json_output_file=args.json_output_file,
        exclude_dir_prefix=args.exclude_dir_prefix,
    )

    scraper = OverallStatsScraper(config)
    scraper.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
