#  `models/`

Model definitions used by **FIRCE** (streaming runtime) and **FIRE** (offline pipeline).

* **`feedforward_binary.py`** — Feed-forward **binary classifier (FNN)** for IDS (Benign vs Attack).
* **`mlp_ce.py`** — **MLP model for Conformal Evaluation (CE)** pipelines (used during CE training/simulation).
* **`torch_device.py`** — Utility helpers to pick the **correct Torch device** (CUDA / MPS / CPU) and set seeds.

---

## When to use which model?

* Use **`feedforward_binary.FeedForwardBinary`** as your primary classifier in runtime or offline training.
* Use **`mlp_ce.MLP_CE`** when training/evaluating models inside CE experiments (ICE/CCE/Approx-CCE/TCE).

Both accept preprocessed numeric feature tensors (e.g., CICFlowMeter features, optionally scaled/PCA’d).

> **Note:** Models typically return **logits**. Apply `torch.softmax` (multi-class) or `torch.sigmoid` (binary with single output) as appropriate for your training/evaluation loop.

---

## Inputs & Preprocessing

* Expect **float32** tensors with shape `[batch, in_features]`.
* Ensure your preprocessing (scaler/PCA, column ordering) **matches** the training configuration used for the model.
* For CE experiments, integrate with the CE wrappers in `core/conformalEval/`.

---

## Files

* **`feedforward_binary.py`** — FNN architecture for binary IDS classification.
* **`mlp_ce.py`** — MLP used in CE pipelines (training/replay under `ce_simulation.py`).
* **`torch_device.py`** — `pick_device()` helper.
* **`__init__.py`** — package marker.


--- 

##  Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)
