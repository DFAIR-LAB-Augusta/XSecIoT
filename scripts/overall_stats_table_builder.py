from __future__ import annotations

import argparse
import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


DEFAULT_DATASET_LABELS: dict[str, str] = {
    "DFAIR": "DFAIR $\\rightarrow$ DFAIR Drift",
    "NB15": "UNSW-NB15 $\\rightarrow$ DFAIR Drift",
    "CIC_UNSW": "CICIDS2018 $\\rightarrow$ DFAIR Drift",
}

DEFAULT_STAT_KEY_MAP: dict[str, str] = {
    "accuracy": "[Classifier Model] Avg Accuracy",
    "precision": "[Classifier Model] Avg Precision",
    "recall": "[Classifier Model] Avg Recall",
    "f1": "[Classifier Model] Avg F1 Score",
    "runtime": "Total simulate time",
    "retrain_count": "Total Drift Detections",
}


class TableConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    json_input_file: Path = Field(default=Path("./logging/ac/overall_stats.json"))
    output_file: Path | None = None
    classifier: str = "feedforward"
    ce_type: str = "approx_cce"
    model_type: str = "binary"
    decimals: int = 4
    require_complete_coverage: bool = False

    dataset_order: tuple[str, ...] = (
        "DFAIR",
        "NB15",
        "CIC_UNSW",
    )

    dataset_labels: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_DATASET_LABELS)
    )
    stat_key_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_STAT_KEY_MAP)
    )

    @field_validator("json_input_file", "output_file", mode="before")
    @classmethod
    def _coerce_optional_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value)

    def resolved_output_file(self) -> Path:
        if self.output_file is not None:
            return self.output_file
        return self.json_input_file.parent / "multi_seed_table_rows.tex"


@dataclass(slots=True, frozen=True)
class GroupedStat:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    stat_key: str
    n: int
    mean_value: float
    std_value: float


@dataclass(slots=True, frozen=True)
class CoverageStat:
    dataset: str
    classifier: str
    ce_type: str
    model_type: str
    expected_total: int
    found_total: int
    missing_pairs: list[str]


class OverallStatsTableBuilder:
    def __init__(self, config: TableConfig) -> None:
        self.config = config

    def run(self) -> Path:
        logger.info("Reading overall stats JSON from %s", self.config.json_input_file)
        payload = self.load_json()
        grouped_stats = self.parse_grouped_stats(payload)
        coverage_stats = self.parse_coverage_stats(payload)
        self.log_available_keys(grouped_stats)
        rows = self.build_rows(grouped_stats, coverage_stats)
        output_path = self.write_output(rows)
        logger.info("Wrote LaTeX table rows to %s", output_path)
        return output_path

    def load_json(self) -> dict[str, Any]:
        if not self.config.json_input_file.exists():
            raise FileNotFoundError(
                f"JSON input file does not exist: {self.config.json_input_file}"
            )

        with self.config.json_input_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if "grouped_summary" not in payload:
            raise ValueError(
                f"Missing 'grouped_summary' in {self.config.json_input_file}"
            )

        return payload

    def parse_grouped_stats(self, payload: dict[str, Any]) -> list[GroupedStat]:
        stats: list[GroupedStat] = []

        for item in payload.get("grouped_summary", []):
            stats.append(
                GroupedStat(
                    dataset=str(item["dataset"]),
                    classifier=str(item["classifier"]),
                    ce_type=str(item["ce_type"]),
                    model_type=str(item["model_type"]),
                    stat_key=str(item["stat_key"]),
                    n=int(item["n"]),
                    mean_value=float(item["mean_value"]),
                    std_value=float(item["std_value"]),
                )
            )

        return stats

    def parse_coverage_stats(self, payload: dict[str, Any]) -> list[CoverageStat]:
        stats: list[CoverageStat] = []

        for item in payload.get("coverage_summary", []):
            stats.append(
                CoverageStat(
                    dataset=str(item["dataset"]),
                    classifier=str(item["classifier"]),
                    ce_type=str(item["ce_type"]),
                    model_type=str(item["model_type"]),
                    expected_total=int(item["expected_total"]),
                    found_total=int(item["found_total"]),
                    missing_pairs=[str(v) for v in item.get("missing_pairs", [])],
                )
            )

        return stats

    def log_available_keys(self, grouped_stats: list[GroupedStat]) -> None:
        filtered = [
            item
            for item in grouped_stats
            if item.classifier == self.config.classifier
            and item.ce_type == self.config.ce_type
            and item.model_type == self.config.model_type
        ]

        logger.debug("Available grouped keys after filtering:")
        for item in sorted(filtered, key=lambda x: (x.dataset, x.stat_key)):
            logger.debug("  dataset=%s stat_key=%s", item.dataset, item.stat_key)

    def build_rows(
        self,
        grouped_stats: list[GroupedStat],
        coverage_stats: list[CoverageStat],
    ) -> list[str]:
        grouped_lookup: dict[tuple[str, str], GroupedStat] = {}
        coverage_lookup: dict[str, CoverageStat] = {}

        for item in grouped_stats:
            if not self._matches_target(item):
                continue
            grouped_lookup[(item.dataset, item.stat_key)] = item

        for item in coverage_stats:
            if not self._matches_target(item):
                continue
            coverage_lookup[item.dataset] = item

        rows: list[str] = []

        for dataset_key in self.config.dataset_order:
            coverage = coverage_lookup.get(dataset_key)
            if self.config.require_complete_coverage and coverage is not None:
                if coverage.found_total != coverage.expected_total:
                    raise ValueError(
                        f"Incomplete coverage for {dataset_key}: "
                        f"{coverage.found_total}/{coverage.expected_total}"
                    )

            dataset_label = self.config.dataset_labels.get(dataset_key, dataset_key)
            accuracy = self._format_stat(grouped_lookup, dataset_key, "accuracy")
            precision = self._format_stat(grouped_lookup, dataset_key, "precision")
            recall = self._format_stat(grouped_lookup, dataset_key, "recall")
            f1 = self._format_stat(grouped_lookup, dataset_key, "f1")
            runtime = self._format_stat(grouped_lookup, dataset_key, "runtime")
            retrain_count = self._format_stat(
                grouped_lookup,
                dataset_key,
                "retrain_count",
            )

            row = (
                f"{dataset_label} \n"
                f"& {accuracy} & {precision} & {recall} "
                f"& {f1} & {runtime} & {retrain_count} \\\\"
            )
            rows.append(row)

        return rows

    def _matches_target(self, item: GroupedStat | CoverageStat) -> bool:
        return (
            item.classifier == self.config.classifier
            and item.ce_type == self.config.ce_type
            and item.model_type == self.config.model_type
        )

    def _format_stat(
        self,
        grouped_lookup: dict[tuple[str, str], GroupedStat],
        dataset_key: str,
        logical_name: str,
    ) -> str:
        stat_key = self.config.stat_key_map[logical_name]
        item = grouped_lookup.get((dataset_key, stat_key))

        if item is None:
            logger.warning(
                "Missing grouped stat for dataset=%s stat_key=%s",
                dataset_key,
                stat_key,
            )
            return "TODO $\\pm$ TODO"

        return self._format_mean_std(item.mean_value, item.std_value, logical_name)

    def _format_mean_std(
        self,
        mean_value: float,
        std_value: float,
        logical_name: str,
    ) -> str:
        if logical_name == "retrain_count":
            return f"{mean_value:.2f} $\\pm$ {std_value:.2f}"

        return (
            f"{mean_value:.{self.config.decimals}f} "
            f"$\\pm$ "
            f"{std_value:.{self.config.decimals}f}"
        )

    def write_output(self, rows: list[str]) -> Path:
        output_path = self.config.resolved_output_file()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("% Auto-generated by scripts/overall_stats_table_builder.py\n")
            handle.write(
                f"% classifier={self.config.classifier}, "
                f"ce_type={self.config.ce_type}, "
                f"model_type={self.config.model_type}\n\n"
            )
            for row in rows:
                handle.write(row)
                handle.write("\n\n")

        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build LaTeX table rows from overall_stats.json."
    )
    parser.add_argument(
        "--json-input-file",
        type=Path,
        default=Path("./logging/ac/overall_stats.json"),
        help="Path to overall_stats.json.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output .tex file path.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="feedforward",
        help="Classifier name to filter on.",
    )
    parser.add_argument(
        "--ce-type",
        type=str,
        default="approx_cce",
        help="CE type to filter on.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="binary",
        help="Model type to filter on.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for mean ± std.",
    )
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help="Fail if any row is missing seed/run coverage.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(debug=args.debug)

    config = TableConfig(
        json_input_file=args.json_input_file,
        output_file=args.output_file,
        classifier=args.classifier,
        ce_type=args.ce_type,
        model_type=args.model_type,
        decimals=args.decimals,
        require_complete_coverage=args.require_complete_coverage,
    )

    builder = OverallStatsTableBuilder(config)
    builder.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())