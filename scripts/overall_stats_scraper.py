from __future__ import annotations

import argparse
import json
import logging
import re

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Iterable, TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class StatMode(str, Enum):
    MAX = 'max'
    MIN = 'min'
    BOTH = 'both'
    ALL = 'all'


class ScraperConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_dir: Path = Field(default=Path('./logging'))
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

    stat_modes: dict[str, StatMode] = {
        'Total simulate time': StatMode.MIN,
        'Total Drift Detections': StatMode.BOTH,
        'Drift Detection Rate': StatMode.BOTH,
        'Average Chunks Between Drift Detections': StatMode.BOTH,
        '[CE Model] Calibrations': StatMode.BOTH,
        '[CE Model] Avg Accuracy': StatMode.MAX,
        '[CE Model] Avg Precision': StatMode.MAX,
        '[CE Model] Avg Recall': StatMode.MAX,
        '[CE Model] Avg F1 Score': StatMode.MAX,
        '[CE Model] Std Accuracy': StatMode.MIN,
        '[Classifier Model] Calibrations': StatMode.BOTH,
        '[Classifier Model] Avg Accuracy': StatMode.MAX,
        '[Classifier Model] Avg Precision': StatMode.MAX,
        '[Classifier Model] Avg Recall': StatMode.MAX,
        '[Classifier Model] Avg F1 Score': StatMode.MAX,
        '[Classifier Model] Std Accuracy': StatMode.MIN,
        'Average Chunk Size': StatMode.ALL,
        'Median Chunk Size': StatMode.ALL,
        'Standard Deviation of Chunk Sizes': StatMode.ALL,
    }

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
class StatEntry:
    subfolder: str
    filename: str
    stat_key: str
    raw_value: str
    numeric_value: float | None


@dataclass(slots=True)
class FileStats:
    subfolder: str
    filename: str
    stat_lines: list[str] = field(default_factory=list)
    entries: list[StatEntry] = field(default_factory=list)

    def add_entry(self, entry: StatEntry) -> None:
        self.entries.append(entry)
        self.stat_lines.append(f'{entry.stat_key}: {entry.raw_value}')


@dataclass(slots=True)
class ParseResult:
    by_subfolder: DefaultDict[str, dict[str, FileStats]] = field(default_factory=lambda: defaultdict(dict))
    by_stat: DefaultDict[str, list[StatEntry]] = field(default_factory=lambda: defaultdict(list))

    def add_file_stats(self, file_stats: FileStats) -> None:
        self.by_subfolder[file_stats.subfolder][file_stats.filename] = file_stats
        for entry in file_stats.entries:
            if entry.numeric_value is not None:
                self.by_stat[entry.stat_key].append(entry)


class OverallStatsScraper:
    CE_SUMMARY_MARKER = '=== CE Model Calibration Summary ==='
    CLASSIFIER_SUMMARY_MARKER = '=== Classifier Model Performance Summary ==='

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.key_value_pattern = re.compile(rf'{re.escape(config.stat_marker)}\s+(.*?):\s+(.*)')

    def run(self) -> tuple[Path, Path]:
        logger.info('Starting scrape from %s', self.config.log_dir)
        result = self.parse_logs()
        log_output_path = self.write_text_output(result)
        json_output_path = self.write_json_output(result)
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
        subfolder = self._relative_subfolder(file_path.parent)
        file_stats = FileStats(subfolder=subfolder, filename=file_path.name)

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
                    subfolder=subfolder,
                    filename=file_path.name,
                    model_section=model_section,
                )
                if entry is not None:
                    file_stats.add_entry(entry)

        return file_stats

    def parse_stat_line(
        self,
        *,
        line: str,
        subfolder: str,
        filename: str,
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
            subfolder=subfolder,
            filename=filename,
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

    def write_text_output(self, result: ParseResult) -> Path:
        output_file = self.config.resolved_output_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open('w', encoding='utf-8') as out:
            self.write_parsed_stats(out, result)
            self.write_best_values(out, result)

        return output_file

    def write_parsed_stats(self, handle: TextIO, result: ParseResult) -> None:
        handle.write('=== Parsed [==OVERALL SIM STATS==] by Subfolder and File ===\n\n')

        for subfolder in sorted(result.by_subfolder):
            handle.write(f'Subfolder: {subfolder}\n')

            for filename in sorted(result.by_subfolder[subfolder]):
                file_stats = result.by_subfolder[subfolder][filename]
                handle.write(f'  File: {filename}\n')

                for stat_line in file_stats.stat_lines:
                    handle.write(f'    {stat_line}\n')

            handle.write('\n')

    def write_best_values(self, handle: TextIO, result: ParseResult) -> None:
        handle.write('=== Best Values by Stat ===\n\n')

        for stat_key in sorted(result.by_stat):
            entries = result.by_stat[stat_key]
            mode = self.config.stat_modes.get(stat_key, StatMode.MAX)
            label = self.display_label(stat_key)

            handle.write(f'{label}:\n')
            self.write_stat_summary(handle, entries, mode)
            handle.write('\n')

    def write_stat_summary(
        self,
        handle: TextIO,
        entries: list[StatEntry],
        mode: StatMode,
    ) -> None:
        if not entries:
            handle.write('  No numeric entries found.\n')
            return

        numeric_entries = [entry for entry in entries if entry.numeric_value is not None]
        if not numeric_entries:
            handle.write('  No numeric entries found.\n')
            return

        match mode:
            case StatMode.MAX:
                max_val = max(entry.numeric_value for entry in numeric_entries)  # type: ignore
                for entry in numeric_entries:
                    if entry.numeric_value == max_val:
                        handle.write(f'  Highest = {entry.raw_value} in {entry.subfolder}/{entry.filename}\n')

            case StatMode.MIN:
                min_val = min(entry.numeric_value for entry in numeric_entries)  # type: ignore
                for entry in numeric_entries:
                    if entry.numeric_value == min_val:
                        handle.write(f'  Lowest = {entry.raw_value} in {entry.subfolder}/{entry.filename}\n')

            case StatMode.BOTH:
                min_val = min(entry.numeric_value for entry in numeric_entries)  # type: ignore
                max_val = max(entry.numeric_value for entry in numeric_entries)  # type: ignore

                handle.write('  Highest:\n')
                for entry in numeric_entries:
                    if entry.numeric_value == max_val:
                        handle.write(f'    {entry.raw_value} in {entry.subfolder}/{entry.filename}\n')

                handle.write('  Lowest:\n')
                for entry in numeric_entries:
                    if entry.numeric_value == min_val:
                        handle.write(f'    {entry.raw_value} in {entry.subfolder}/{entry.filename}\n')

            case StatMode.ALL:
                sorted_entries = sorted(
                    numeric_entries,
                    key=lambda entry: entry.numeric_value if entry.numeric_value is not None else float('inf'),
                )
                for entry in sorted_entries:
                    handle.write(f'  {entry.raw_value} in {entry.subfolder}/{entry.filename}\n')

            case _:
                raise ValueError(f'Unknown stat mode: {mode}')

    def write_json_output(self, result: ParseResult) -> Path:
        output_file = self.config.resolved_json_output_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        payload = self.build_json_payload(result)

        with output_file.open('w', encoding='utf-8') as out:
            json.dump(payload, out, indent=2, sort_keys=True)

        return output_file

    def build_json_payload(self, result: ParseResult) -> dict[str, Any]:
        return {
            'config': {
                'log_dir': str(self.config.log_dir),
                'output_file': str(self.config.resolved_output_file()),
                'json_output_file': str(self.config.resolved_json_output_file()),
                'stat_marker': self.config.stat_marker,
                'exclude_dir_prefix': self.config.exclude_dir_prefix,
                'log_suffix': self.config.log_suffix,
            },
            'parsed': self._build_parsed_json(result),
            'best_values': self._build_best_values_json(result),
        }

    def _build_parsed_json(self, result: ParseResult) -> dict[str, dict[str, Any]]:
        parsed: dict[str, dict[str, Any]] = {}

        for subfolder in sorted(result.by_subfolder):
            parsed[subfolder] = {}
            for filename in sorted(result.by_subfolder[subfolder]):
                file_stats = result.by_subfolder[subfolder][filename]
                parsed[subfolder][filename] = {
                    'stat_lines': file_stats.stat_lines,
                    'entries': [asdict(entry) for entry in file_stats.entries],
                }

        return parsed

    def _build_best_values_json(self, result: ParseResult) -> dict[str, Any]:
        best_values: dict[str, Any] = {}

        for stat_key in sorted(result.by_stat):
            entries = result.by_stat[stat_key]
            mode = self.config.stat_modes.get(stat_key, StatMode.MAX)
            best_values[stat_key] = {
                'display_label': self.display_label(stat_key),
                'mode': mode.value,
                'results': self._summarize_entries(entries, mode),
            }

        return best_values

    def _summarize_entries(
        self,
        entries: list[StatEntry],
        mode: StatMode,
    ) -> dict[str, Any]:
        numeric_entries = [entry for entry in entries if entry.numeric_value is not None]
        if not numeric_entries:
            return {'message': 'No numeric entries found.'}

        match mode:
            case StatMode.MAX:
                max_val = max(entry.numeric_value for entry in numeric_entries)  # type: ignore
                matches = [asdict(entry) for entry in numeric_entries if entry.numeric_value == max_val]
                return {'highest': matches}

            case StatMode.MIN:
                min_val = min(entry.numeric_value for entry in numeric_entries)  # type: ignore
                matches = [asdict(entry) for entry in numeric_entries if entry.numeric_value == min_val]
                return {'lowest': matches}

            case StatMode.BOTH:
                min_val = min(entry.numeric_value for entry in numeric_entries)  # type: ignore
                max_val = max(entry.numeric_value for entry in numeric_entries)  # type: ignore
                highest = [asdict(entry) for entry in numeric_entries if entry.numeric_value == max_val]
                lowest = [asdict(entry) for entry in numeric_entries if entry.numeric_value == min_val]
                return {
                    'highest': highest,
                    'lowest': lowest,
                }

            case StatMode.ALL:
                sorted_entries = sorted(
                    numeric_entries,
                    key=lambda entry: entry.numeric_value if entry.numeric_value is not None else float('inf'),
                )
                return {
                    'ordered': [asdict(entry) for entry in sorted_entries],
                }

            case _:
                raise ValueError(f'Unknown stat mode: {mode}')

    def display_label(self, stat_key: str) -> str:
        if stat_key in self.config.chunk_stats:
            return f'Adaptive Chunker {stat_key}'
        return stat_key

    def _relative_subfolder(self, directory: Path) -> str:
        relative = directory.relative_to(self.config.log_dir)
        return relative.as_posix() if relative.parts else '.'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Scrape FIRCE overall stats from log files.')
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('./logging'),
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
