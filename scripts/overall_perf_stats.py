# src/utils/perf_stats_ce_only_plots.py
"""
Classifier dashboards: 5x4 grid of CE + Classifier metrics per classifier.

- Scans: ./logging/DFAIR/acDFAIR (recursive, overridable via --log-dir)
- Finds lines containing:
    INFO __main__: Full performance stats: perf_stats = PerformanceStats(...)
- Parses BOTH:
    ce_stats = ModelStats(accuracies=[...], precisions=[...], recalls=[...], f1s=[...])
    classifier_stats = ModelStats(accuracies=[...], precisions=[...], recalls=[...], f1s=[...])

Outputs (per classifier):
  - <out-dir>/<classifier>_ce_grid.png      # 5x4 grid: CE types across, metrics down (2 lines per subplot)
  - <out-dir>/<classifier>_ce_metrics.csv   # long-form CSV: series, run, ce_type, metric, step, value

Usage:
    uv run python -m src.utils.perf_stats_ce_only_plots \
        --log-dir ./logging/DFAIR/acDFAIR \
        --out-dir ./logging/acDFAIRGraphs \
        --verbose
"""

from __future__ import annotations

import argparse
import csv
import logging
import re

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

matplotlib.use("Agg")


DEFAULT_LOG_DIR = Path("./logging/ac_UNSW")
DEFAULT_OUT_DIR = Path("./logging/ac_UNSW_graphs")

PERF_MARKER = re.compile(
    r"INFO\s+__main__:\s+Full performance stats:\s+perf_stats\s*=\s*PerformanceStats\(",
    re.IGNORECASE,
)

CE_BLOCK = re.compile(
    r"ce_stats\s*=\s*ModelStats\((?P<inside>.*?)\)", re.DOTALL)
CLF_BLOCK = re.compile(
    r"classifier_stats\s*=\s*ModelStats\((?P<inside>.*?)\)", re.DOTALL)

LIST_PATTERNS = {
    "accuracies": re.compile(r"accuracies\s*=\s*\[([^\]]*)\]"),
    "precisions": re.compile(r"precisions\s*=\s*\[([^\]]*)\]"),
    "recalls":    re.compile(r"recalls\s*=\s*\[([^\]]*)\]"),
    "f1s":        re.compile(r"f1s\s*=\s*\[([^\]]*)\]"),
}

MODELVAR_RE = re.compile(
    r"\bmodelVariant\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
CETYPE_RE = re.compile(r"\bceType\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)

CLASSIFIER_CANON = {
    "svm": ["svm", "svc"],
    "dt": ["dt", "decisiontree", "decision_tree", "decision-tree"],
    "rf": ["rf", "randomforest", "random_forest", "random-forest"],
    "xgb": ["xgb", "xgboost", "xgbclassifier"],
    "knn": ["knn", "k-nearest", "kneighbors", "k-nearest-neighbors", "k-nearest-neighbours"],
    "logreg": ["logreg", "logistic", "logisticregression", "lr"],
    "mlp": ["mlp", "mlpclassifier", "neural", "nn"],
    "sgd": ["sgd"],
    "nb": ["naive_bayes", "nb", "gaussian_nb", "gaussiannb"],
    "lgbm": ["lgbm", "lightgbm"],
    "ffn": ["ffn", "feedforward"]
}

CE_CANON = {
    "approx-cce": ["approx-cce", "approxcce", "approx_cce", "approxcce", "approx-cce".upper()],
    "cce": ["cce"],
    "ice": ["ice"],
    "tce": ["tce"],
    "none": ["none", "noce", "no-ce", "no_ce", "baseline", "null"],
    "approx-tce": ["approx-tce", "approxtce", "approx_tce", "approxtce".upper()],
}

CE_ORDER = ["approx-cce", "cce", "ice", "tce", "none"]
METRICS = [
    ("accuracies", "Accuracy"),
    ("precisions", "Precision"),
    ("recalls", "Recall"),
    ("f1s", "F1 Score"),
]


def _parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[float] = []
    for tok in s.split(","):
        t = tok.strip().rstrip("%")
        if not t or t.lower() == "nan":
            continue
        try:
            out.append(float(t))
        except ValueError:
            logging.debug("Skipping token not parseable as float: %r", tok)
    return out


def _parse_modelstats_block(inner: str) -> Dict[str, List[float]]:
    """
    Parse one ModelStats(...) inner string into metric lists.
    """
    out: Dict[str, List[float]] = {}
    for k, pat in LIST_PATTERNS.items():
        mm = pat.search(inner)
        out[k] = _parse_float_list(mm.group(1)) if mm else []
    return out


def _parse_both_series(line: str) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """
    Parse both CE and Classifier series from the PerformanceStats(...) line.
    Returns:
        {'ce': {'accuracies': [...], ...}, 'classifier': {'accuracies': [...], ...}}
    """
    if not PERF_MARKER.search(line):
        return None

    ce_series: Dict[str, List[float]] = {}
    clf_series: Dict[str, List[float]] = {}

    m_ce = CE_BLOCK.search(line)
    if m_ce:
        ce_series = _parse_modelstats_block(m_ce.group("inside"))
    else:
        logging.warning("Perf stats found, but ce_stats block missing.")

    m_clf = CLF_BLOCK.search(line)
    if m_clf:
        clf_series = _parse_modelstats_block(m_clf.group("inside"))
    else:
        logging.warning(
            "Perf stats found, but classifier_stats block missing.")

    if not ce_series and not clf_series:
        return None
    return {"ce": ce_series, "classifier": clf_series}


def _canonicalize(token: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    t = token.lower()
    for canon, synonyms in mapping.items():
        for s in synonyms:
            if s in t:
                return canon
        if t == canon:
            return canon
    return None


def _infer_classifier_ce_from_content_or_path(fpath: Path) -> Tuple[str, str]:
    classifier, ce_type = None, None

    try:
        with fpath.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                mv = MODELVAR_RE.search(line)
                if mv and not classifier:
                    classifier = _canonicalize(mv.group(1), CLASSIFIER_CANON)
                ct = CETYPE_RE.search(line)
                if ct and not ce_type:
                    ce_type = _canonicalize(ct.group(1), CE_CANON)
                if classifier and ce_type:
                    break
    except Exception:
        logging.exception("Failed reading %s while inferring labels.", fpath)

    path_str = fpath.as_posix().lower()
    if not classifier:
        for canon, syns in CLASSIFIER_CANON.items():
            if any(s in path_str for s in syns) or canon in path_str:
                classifier = canon
                break
    if not ce_type:
        for canon, syns in CE_CANON.items():
            if any(s in path_str for s in syns) or canon in path_str:
                ce_type = canon
                break

    return (classifier or "unknown", ce_type or "unknown")


def _scan_logs_group_by_classifier(
    log_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]]:
    """
    Build:
        grouped[classifier][ce_type][run_label] = {
            'ce': {'accuracies': [...], 'precisions': [...], 'recalls': [...], 'f1s': [...]},
            'classifier': {...}
        }
    """
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]
                  ] = defaultdict(lambda: defaultdict(dict))
    files = list(log_dir.rglob("*.log"))
    if not files:
        logging.error("No .log files found under %s", log_dir)
        return grouped

    for fpath in files:
        run_rel = fpath.relative_to(log_dir).as_posix()
        last_both: Optional[Dict[str, Dict[str, List[float]]]] = None
        try:
            with fpath.open("r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    line = raw.strip()
                    if "PerformanceStats(" not in line:
                        continue
                    both = _parse_both_series(line)
                    if both is not None:
                        last_both = both
        except Exception:
            logging.exception("Failed reading %s", fpath)
            continue

        if last_both is None:
            logging.warning("No CE/Classifier series parsed in %s", run_rel)
            continue

        classifier, ce_type = _infer_classifier_ce_from_content_or_path(fpath)
        grouped[classifier][ce_type][run_rel] = last_both
        logging.info("Parsed CE+Classifier for %s | classifier=%s, ce=%s",
                     run_rel, classifier, ce_type)

    if not grouped:
        logging.error("Found logs, but no series could be parsed.")
    else:
        logging.info("Parsed series for %d classifier groups.", len(grouped))

    return grouped


def _plot_classifier_grid(
    classifier: str,
    runs_by_ce: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_png: Path,
) -> None:
    n_rows = len(METRICS)
    n_cols = len(CE_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(
        4.0 * n_cols, 3.0 * n_rows), squeeze=False)

    metric_is_percent = {mk: False for mk, _ in METRICS}
    for ce_key in CE_ORDER:
        bucket = runs_by_ce.get(ce_key, {})
        for _run_label, series_pair in bucket.items():
            if not isinstance(series_pair, dict):
                continue
            for series_name in ("ce", "classifier"):
                mdict = series_pair.get(series_name, {})
                if not isinstance(mdict, dict):
                    continue
                for metric_key, _metric_label in METRICS:
                    ys = mdict.get(metric_key, [])
                    if ys:
                        if max(ys) > 1.05:
                            metric_is_percent[metric_key] = True

    for col_idx, ce_key in enumerate(CE_ORDER):
        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            bucket = runs_by_ce.get(ce_key, {})
            any_data = False

            for _run_label, series_pair in bucket.items():
                if not isinstance(series_pair, dict):
                    continue
                ce_series = series_pair.get("ce", {})
                clf_series = series_pair.get("classifier", {})

                if isinstance(ce_series, dict):
                    y = ce_series.get(metric_key, [])
                    if y:
                        any_data = True
                        y_plot = [
                            (v / 100.0) if metric_is_percent[metric_key] else v for v in y]
                        x = list(range(1, len(y_plot) + 1))
                        ax.plot(x, y_plot, linestyle="-",
                                marker=".", linewidth=1, label=None)

                if isinstance(clf_series, dict):
                    y2 = clf_series.get(metric_key, [])
                    if y2:
                        any_data = True
                        y2_plot = [
                            (v / 100.0) if metric_is_percent[metric_key] else v for v in y2]
                        x2 = list(range(1, len(y2_plot) + 1))
                        ax.plot(x2, y2_plot, linestyle="--",
                                marker="x", linewidth=1, label=None)

            if not any_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=9, alpha=0.7, transform=ax.transAxes)
            else:
                ax.set_ylim(0, 1.0)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Calibration step")
            if col_idx == 0:
                ylabel = metric_label + \
                    (" (proportion of 1.0)" if metric_is_percent[metric_key] else "")
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(ce_key)

            legend_handles = [
                Line2D([0], [0], linestyle="-", marker=".",
                       linewidth=1, label="CE"),
                Line2D([0], [0], linestyle="--", marker="x",
                       linewidth=1, label="Classifier"),
            ]
            ax.legend(handles=legend_handles, loc="best",
                      fontsize="small", frameon=False)
            ax.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logging.info("Wrote %s", out_png)


def _write_classifier_csv(
    classifier: str,
    # ce_type -> run_label -> {'ce':..., 'classifier':...}
    runs_by_ce: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_csv: Path,
) -> None:
    """
    Long-form CSV columns: classifier, series, run, ce_type, metric, step, value
    where 'series' ∈ {'ce','classifier'}
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["classifier", "series", "run",
                   "ce_type", "metric", "step", "value"])
        for ce_key in CE_ORDER:
            bucket = runs_by_ce.get(ce_key, {})
            for run_label, series_pair in bucket.items():
                for series_name in ("ce", "classifier"):
                    mdict = series_pair.get(series_name, {})
                    if not isinstance(mdict, dict):
                        continue
                    for metric_key, _metric_label in METRICS:
                        y = mdict.get(metric_key, [])
                        for i, v in enumerate(y, start=1):
                            w.writerow([classifier, series_name,
                                       run_label, ce_key, metric_key, i, v])
    logging.info("Wrote %s", out_csv)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classifier CE+Classifier dashboards (5x4 grid per classifier).")
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                    help="Directory to scan for logs.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Directory to write images/CSVs.")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable debug logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log_dir = args.log_dir
    out_dir = args.out_dir

    grouped = _scan_logs_group_by_classifier(log_dir)
    if not grouped:
        logging.error("No series found; exiting.")
        return

    for classifier, ce_to_runs in grouped.items():
        extras = [k for k in ce_to_runs.keys() if k not in CE_ORDER]
        if extras:
            logging.warning(
                "Classifier '%s' has CE types not in CE_ORDER and will be omitted: %s",
                classifier, extras
            )

        png_path = out_dir / f"{classifier}_ce_grid.png"
        csv_path = out_dir / f"{classifier}_ce_metrics.csv"

        _plot_classifier_grid(classifier, ce_to_runs, png_path)  # type: ignore
        _write_classifier_csv(classifier, ce_to_runs, csv_path)  # type: ignore

    logging.info("Done.")


if __name__ == "__main__":
    main()
