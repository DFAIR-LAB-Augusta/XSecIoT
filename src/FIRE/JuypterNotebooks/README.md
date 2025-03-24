# 📂 JuypterNotebooks/

This directory contains the original Jupyter notebooks used in the development of the **FIRE** framework. The notebooks are structured into three stages:

---

## 1. ✏️ Preprocessing

These notebooks handle the initial transformation of raw network flow data into ML-ready format:

* `1_IoTAttacks__runtime_latency_parallel_processing.ipynb`
* `2_Model_training_on_Session_Time based_sliding_Aggregation.ipynb`

They implement aggregation strategies (e.g., time-windowed flow slicing), encoding, and other transformations needed prior to training.

---

## 2. 🎯 Model Training & Evaluation

Located in the `3/` subdirectory, these notebooks focus on training and evaluating various supervised learning models. Each file corresponds to a distinct ML method:

* `model_K_Nearest_Neighbor.ipynb`
* `model_XGBoost_modified.ipynb`
* `model_decision_tree.ipynb`
* `model_feed_forward.ipynb`
* `model_random_forest.ipynb`
* `model_support_vector_machine.ipynb`

Each notebook includes preprocessing references, training code, evaluation metrics, and (where applicable) visualization.

---

## 📓 Usage

These notebooks are preserved for:

* Reference to the development process
* Visualization and experimentation
* Reproducing FIRE's model benchmarks

They are no longer part of the runtime system but remain useful for manual experimentation.

---

For questions, contact **[sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)**.
