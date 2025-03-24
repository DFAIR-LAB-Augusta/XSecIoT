# 🔥 FIRCE (XSecIoT): Streaming Conformal Evaluation for IoT IDS

**FIRCE** (Framework for **I**ntrusion **R**esponse and **C**onformal **E**valuation) is the streaming half of **XSecIoT**. It ingests flow records, performs ML classification, detects concept drift with Conformal Evaluators (ICE/CCE/Approx-CCE/TCE), and triggers adaptive retraining with rolling logs.

This repository also contains **FIRE** (offline preprocessing/modeling/simulation), but the primary entrypoint for users is **FIRCE**. The FIRE repo can be found in its state at publication in the `FIRE_bkp` branch on this repository.

---

## 📦 What You Get

* **Streaming IDS pipeline** (`src/core/streaming_pipeline.py`) with:

  * Conformal Evaluation (ICE, CCE, Approx-CCE, TCE)
  * Adaptive chunking and adaptive significance control
  * Rolling log + optional circular logger
* **Batteries-included runner**: `./src/core/run_sim.sh`
* **Configurable CE settings**: `src/core/conformalEval/conformal_config.toml`
* **Artifacts and outputs**:

  * `binary_models/` and `multiclass_models/` (saved models)
  * `logging/` (run logs, performance summaries)
  * `datasets/` (your input data)

---

## 🗂️ Project Layout (focused on FIRCE)

```
XSecIoT/
├── datasets/                 # ← Input datasets (CSV flows, calibration sets)
├── binary_models/            # ← Output: saved binary classifiers
├── multiclass_models/        # ← Output: saved multiclass classifiers
├── logging/                  # ← Output: run logs & performance stats
├── src/
│   ├── core/
│   │   ├── ce_simulation.py
│   │   ├── streaming_pipeline.py
│   │   ├── run_sim.sh        # ← Main launcher for FIRCE simulation
│   │   ├── conformalEval/
│   │   │   ├── approx_cce.py
│   │   │   ├── cce.py
│   │   │   ├── ice.py
│   │   │   ├── tce.py
│   │   │   ├── utils.py
│   │   │   └── conformal_config.toml
│   │   ├── models/           # MLP/FFN CE models + device helpers
│   │   ├── adaptive_chunking.py
│   │   ├── adaptive_sig_ctlr.py
│   │   ├── circular_logger.py
│   │   ├── rolling_csv.py
│   │   └── perf_stats.py
│   ├── FIRE/                 # Offline pipeline (preprocess/train/simulate)
│   └── utils/                # Python utils for processing log output
│       ├── labeling.py
│       ├── merge.py
│       ├── overall_perf_stats.py
│       └── overall_stats_scraper.py
├── tests/
├── pyproject.toml
└── uv.lock
```


> **Directory semantics:**
>
> * `datasets/` is an **input** directory.
> * `binary_models/`, `multiclass_models/`, and `logging/` are **output** directories.

---

## ⚙️ Requirements

* Python 3.10+
* [UV](https://docs.astral.sh/uv/) (fast Python package/environment manager)

---

## 🚀 Quick Start (FIRCE)

1. **Install dependencies**

```bash
uv sync
```

2. **Place your CSV flows** in `datasets/`

   * Minimum: one **calibration**/train CSV and one **stream** CSV.
   * Example:

     ```
     datasets/
     ├── CETrain/
     │   └── combined_data.csv
     └── CEFlows/
         └── your_stream_flows.csv
     ```

3. **Run FIRCE**

```bash
./src/core/run_sim.sh
```

That’s it—FIRCE will load from `datasets/`, run the streaming CE pipeline, and write artifacts/metrics to `logging/` and trained models to `binary_models/` or `multiclass_models/` as applicable.

---

## 🔧 Configuration Tips

* **Conformal Evaluators & thresholds:** edit `src/core/conformalEval/conformal_config.toml` to switch CE type (ICE/CCE/Approx-CCE/TCE), calibration window sizes, p-value thresholds, etc.
* **Adaptive behavior:** tune `adaptive_chunking.py` and `adaptive_sig_ctlr.py` parameters if you need different responsiveness.
* **Hardware selection:** `src/core/models/torch_device.py` auto-selects device; override via env var if needed.

---

## ⚠️ Disclaimer

This is **research code** for academic use. Validate thoroughly before any production deployment.

---

## 📢 Contact

- Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
- Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
- Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
- Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
- Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)

🔐 Securing IoT, one flow at a time.
