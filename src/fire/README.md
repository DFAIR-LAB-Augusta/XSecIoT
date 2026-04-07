# `fire/`

This directory contains the **FIRE (Fog-based Intrusion Detection Framework for Real-time security in IoT Environments)** implementation used for **batch-based intrusion detection experiments**.

FIRE serves as the **baseline ML pipeline** within the broader FIRCE ecosystem, providing:
- Dataset preprocessing
- Model training
- Batch simulation and evaluation

Unlike FIRCE, FIRE does **not** include streaming or conformal evaluation. It is primarily used for **offline experimentation and benchmarking**.

---

## Directory Layout

```text
fire/
├── main.py            # Entrypoint for FIRE pipeline execution
├── preprocessing.py   # Dataset preprocessing and feature engineering
├── models.py          # ML model definitions (training + inference)
├── simulations.py     # Batch simulation and evaluation logic
└── README.md
```

---

## Components

| File                   | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| **`main.py`**          | Orchestrates the FIRE pipeline (preprocess → train → evaluate) |
| **`preprocessing.py`** | Cleans and transforms raw datasets into model-ready features   |
| **`models.py`**        | Defines classifiers used for intrusion detection               |
| **`simulations.py`**   | Runs evaluation experiments on prepared datasets               |

---

## Execution Flow

1. **Preprocessing**

   * Load dataset
   * Clean and normalize features
   * Prepare labels

2. **Training**

   * Train supervised ML model(s)
   * Optionally apply feature transformations

3. **Evaluation**

   * Run simulations on test data
   * Compute metrics (accuracy, precision, recall, F1)

---

## Usage

From the repository root:

```bash
uv run python -m src.fire.main <dataset_path>
```

Example:

```bash
uv run python -m src.fire.main \
    datasets/DFAIR/combined_data_with_okpVacc_modified.csv
```

Output (logs, metrics) will be printed to stdout or handled via configured logging.

---

## Relationship to FIRCE

FIRE is the **precursor and baseline** to FIRCE:

| FIRE                      | FIRCE                                  |
| ------------------------- | -------------------------------------- |
| Batch processing          | Streaming + simulation pipelines       |
| Static model evaluation   | Adaptive retraining + drift detection  |
| No drift detection        | Conformal + CADE-based drift detection |
| No statistical guarantees | CE-based statistical guarantees        |

FIRE is primarily used to:

* Benchmark model performance
* Generate baseline results
* Provide training components reused in FIRCE

---

## Notes

* All original Jupyter notebooks have been removed in favor of a **pure Python implementation**.
* The codebase has been streamlined for integration into FIRCE workflows.
* This module is intentionally minimal and stable compared to FIRCE’s experimental components.

---

## Contact

Seth Barrett
[https://github.com/sethbarrett50](https://github.com/sethbarrett50)
[sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)

Bradley Boswell
[https://github.com/bradleyboswell](https://github.com/bradleyboswell)
[brboswell@augusta.edu](mailto:brboswell@augusta.edu)

Swarnamugi Rajaganapathy, PhD
[https://github.com/swarna6384](https://github.com/swarna6384)
[swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)

Lin Li, PhD
[https://github.com/linli786](https://github.com/linli786)
[lli1@augusta.edu](mailto:lli1@augusta.edu)

Gokila Dorai, PhD
[https://github.com/gdorai](https://github.com/gdorai)
[gdorai@augusta.edu](mailto:gdorai@augusta.edu)
