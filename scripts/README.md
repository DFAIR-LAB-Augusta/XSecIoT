# 🧰 `utils/` — Log & Metrics Utilities

Helper scripts for inspecting, labeling, merging, and summarizing outputs produced by **FIRCE/FIRE**.
All scripts can be executed with **UV**.

> **Conventions**
>
> * **Inputs** typically come from `datasets/`.
> * **Outputs** (logs, artifacts, summaries) are written under `logging/`, with trained models stored in `binary_models/` or `multiclass_models/`.
> * Run from the **repo root** with `uv run --project . <path-to-script> [args]`.
> * Many scripts expose CLI flags via `-h/--help`.

---

## Files

* **`labeling.py`**
  Utilities for applying/adjusting labels on flow logs or merged CSVs (e.g., adding ground-truth, correcting labels post hoc).

* **`merge.py`**
  Safe CSV merging for multi-run or sharded outputs (e.g., combine `logging/run_*/*.csv` into a single analysis table).

* **`overall_perf_stats.py`**
  Aggregates run-level metrics (accuracy, F1, runtime, etc.) across experiments; emits consolidated performance summaries.

* **`overall_stats_scraper.py`**
  Parses logs to extract standardized “Full performance stats:” blocks and normalizes them into a tabular CSV.

---

## Quick Start

From the **repository root**:

```bash
uv sync
```

Then, for any script (examples below):

```bash
# Show usage/flags (if implemented via argparse/click)
uv run --project . utils/overall_perf_stats.py --help
```

> If you prefer running from within this directory, you can also use:
>
> ```bash
> uv run --project .. overall_perf_stats.py --help
> ```

---

## Common Workflows (Examples)

> Adjust paths/patterns to match your run layout under `logging/`.

### 1) Merge CSV shards from multiple runs

```bash
uv run --project . utils/merge.py \
  --input "logging/run_*/flows_*.csv" \
  --output logging/merged/flows_merged.csv
```

### 2) Scrape “Full performance stats:” blocks from logs

```bash
uv run --project . utils/overall_stats_scraper.py \
  --input "logging/run_*/perf.log" \
  --output logging/summary/perf_stats_raw.csv
```

### 3) Build an overall performance table

```bash
uv run --project . utils/overall_perf_stats.py \
  --input logging/summary/perf_stats_raw.csv \
  --output logging/summary/perf_overall.csv
```

### 4) Apply or correct labels on merged flows

```bash
uv run --project . utils/labeling.py \
  --input logging/merged/flows_merged.csv \
  --map datasets/label_maps/labels.json \
  --output logging/labeled/flows_labeled.csv
```

---

## Tips

* Keep intermediate artifacts under `logging/` (e.g., `logging/merged/`, `logging/summary/`, `logging/labeled/`) to avoid polluting input data in `datasets/`.
* For reproducibility, commit small summary CSVs (e.g., `perf_overall.csv`) but **do not** commit large raw logs.
* If a script fails due to missing columns, ensure your upstream pipeline produced the expected headers (FIRCE/FIRE versions should match).


--- 

## 📢 Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)
