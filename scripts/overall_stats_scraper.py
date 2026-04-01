# src/utils/overall_stats_scraper
import logging
import os
import re

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

LOG_DIR = Path("./logging")
OUTPUT_FILE = LOG_DIR / "overall_stats.log"
STAT_MARKER = "[==OVERALL SIM STATS==]"
EXCLUDE_DIR_PREFIX = "old"
CHUNK_STATS = {
    "Average Chunk Size",
    "Median Chunk Size",
    "Standard Deviation of Chunk Sizes",
}

stats_by_run: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
stat_values: Dict[str, List[Tuple[str, str, float, str]]] = defaultdict(list)

stat_modes = {
    "Total simulate time": "min",
    "Total Drift Detections": "both",
    "Drift Detection Rate": "both",
    "Average Chunks Between Drift Detections": "both",
    "[CE Model] Calibrations": "both",
    "[CE Model] Avg Accuracy": "max",
    "[CE Model] Avg Precision": "max",
    "[CE Model] Avg Recall": "max",
    "[CE Model] Avg F1 Score": "max",
    "[CE Model] Std Accuracy": "min",
    "[Classifier Model] Calibrations": "both",
    "[Classifier Model] Avg Accuracy": "max",
    "[Classifier Model] Avg Precision": "max",
    "[Classifier Model] Avg Recall": "max",
    "[Classifier Model] Avg F1 Score": "max",
    "[Classifier Model] Std Accuracy": "min",
    "Average Chunk Size": "all",
    "Median Chunk Size": "all",
    "Standard Deviation of Chunk Sizes": "all",
}


key_value_pattern = re.compile(rf"{re.escape(STAT_MARKER)}\s+(.*?):\s+(.*)")

for subdir, _, files in os.walk(LOG_DIR):
    if Path(subdir).name.startswith(EXCLUDE_DIR_PREFIX):
        continue
    for file in files:
        if file.endswith(".log"):
            file_path = Path(subdir) / file
            subfolder = Path(subdir).relative_to(LOG_DIR).as_posix()
            model_section = None
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=== CE Model Calibration Summary ===" in line:
                            model_section = "CE Model"
                        elif "=== Classifier Model Performance Summary ===" in line:
                            model_section = "Classifier Model"
                        elif STAT_MARKER in line:
                            match = key_value_pattern.search(line)
                            if match:
                                stat_key, stat_value = match.group(1).strip(), match.group(2).strip()

                                if stat_key in CHUNK_STATS:
                                    full_key = stat_key  # no prefix
                                else:
                                    prefix = f"[{model_section}]" if model_section else ""
                                    full_key = f"{prefix} {stat_key}".strip()

                                full_stat_line = f"{full_key}: {stat_value}"
                                stats_by_run[subfolder][file].append(full_stat_line)

                                try:
                                    val = float(stat_value.strip('%s'))
                                    stat_values[full_key].append((subfolder, file, val, stat_value))
                                except ValueError:
                                    continue
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

LOG_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("=== Parsed [==OVERALL SIM STATS==] by Subfolder and File ===\n\n")
    for subfolder, files in stats_by_run.items():
        out.write(f"Subfolder: {subfolder}\n")
        for filename, stats in files.items():
            out.write(f"  File: {filename}\n")
            for stat in stats:
                out.write(f"    {stat}\n")
        out.write("\n")

    out.write("=== Best Values by Stat ===\n\n")
    for stat_key, entries in stat_values.items():
        mode = stat_modes.get(stat_key, "max")
        label = stat_key
        if stat_key in {
            "Average Chunk Size",
            "Median Chunk Size",
            "Standard Deviation of Chunk Sizes",
        }:
            label = f"Adaptive Chunker {stat_key}"
        out.write(f"{label}:\n")

        match (mode):
            case "max":
                max_val = max(entries, key=lambda x: x[2])[2]
                for s, f, v, raw in entries:
                    if v == max_val:
                        out.write(f"  Highest = {raw} in {s}/{f}\n")

            case "min":
                min_val = min(entries, key=lambda x: x[2])[2]
                for s, f, v, raw in entries:
                    if v == min_val:
                        out.write(f"  Lowest = {raw} in {s}/{f}\n")

            case "both":
                min_val = min(entries, key=lambda x: x[2])[2]
                max_val = max(entries, key=lambda x: x[2])[2]
                out.write("  Highest:\n")
                for s, f, v, raw in entries:
                    if v == max_val:
                        out.write(f"    {raw} in {s}/{f}\n")
                out.write("  Lowest:\n")
                for s, f, v, raw in entries:
                    if v == min_val:
                        out.write(f"    {raw} in {s}/{f}\n")

            case "all":
                sorted_entries = sorted(entries, key=lambda x: x[2])
                for s, f, v, raw in sorted_entries:
                    out.write(f"  {raw} in {s}/{f}\n")

            case _:
                raise ValueError(f"Unknown mode '{mode}' for stat '{stat_key}'")
        out.write("\n")