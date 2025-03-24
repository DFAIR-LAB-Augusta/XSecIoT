# 🔢 FIRE\_codebase/

This directory contains the original implementation of the **FIRE (Fog-based Intrusion detection framework for Real-time security in IoT Environments)** framework. It includes scripts and notebooks used for data preprocessing, model training, and simulation-based evaluation of intrusion detection models.

---

## 🔧 Core Files

| File                   | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `main.py`              | Entrypoint for executing the full FIRE ML pipeline              |
| `preprocessing.py`     | Preprocesses the DFAIR dataset for model training               |
| `models.py`            | Defines supervised learning models used in classification tasks |
| `simulations.py`       | Executes simulations and evaluations for the DFAIR dataset      |
| `simulationsunsw.py`   | Executes simulations and evaluations for the UNSW dataset       |

---

## 📂 JuypterNotebooks/

This folder contains the original Jupyter notebooks used during the development of the FIRE framework. These include exploratory experiments, model training iterations, and performance evaluations.

Example notebooks:

* `1_IoTAttacks__runtime_latency_parallel_processing.ipynb`
* `2_Model_training_on_Session_Time_based_sliding_Aggregation.ipynb`
* `3/model_XGBoost_modified.ipynb`, etc.

These are preserved for reference, experimentation, and visualization.

---

## 🔁 Integration

This package can be executed independently or used within larger systems like `core`. You can run the entire FIRE pipeline using:

```bash
python -m FIRE.main ./datasets/DFAIR/combined_data_with_okpVacc_modified.csv > output.txt 2>&1
```

---

💡 Use this codebase for batch experimentation, model benchmarking, and evaluation of dataset-specific performance.

## 📢 Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)