# 📐 Conformal Evaluation Module

Implements a suite of **Conformal Evaluators** for uncertainty-aware detection of **concept drift** using nonconformity scores and conformal **p-values**. Supported evaluators:

* **ICE** — Inductive Conformal Evaluation
* **CCE** — Cross-Conformal Evaluation
* **TCE** — Transductive Conformal Evaluation
* **Approx-TCE** — Approximate TCE (streamlined for streaming)
* **Approx-CCE** — *Our proposed* lightweight CCE approximation for streaming

Use the factory in **`conformal_evaluators.py`** to instantiate by name:
`'ice' | 'cce' | 'tce' | 'approx_tce' | 'approx_cce'`.

---

## 🔢 Core Concepts

### 1) Nonconformity Measure (NCM)

Given a **bag** (B) of past examples:

![Nonconformity](../../../assets/ncm1.svg)

and a new example (z), define a nonconformity score (\alpha(z,|,B))

![Nonconformity](../../../assets/ncm2.svg)

which quantifies how “strange” (z) is relative to (B). A common instantiation:

![Nonconformity](../../../assets/ncm3.svg)

where (\hat z(B)) is a predictor fit on (B) (e.g., class prototype / decision function), and (d(\cdot,\cdot)) is a dissimilarity metric.

---

### 2) Conformal p-values

For candidate class (y) and test point (z^*), let (\alpha_y(\cdot)) be the class-conditional NCM:

![P-Value Computation](../../../assets/pComp1.svg)

The **conformal p-value** is

![P-Value Computation](../../../assets/pComp2.svg)

Small p-values indicate **atypicality / drift**:

![Inline](../../../assets/inline1.svg) small  ⇒  ![Inline](../../../assets/inline2.svg) likely out-of-distribution.

---

### 3) Thresholds & Drift Decision

Choose per-class thresholds (\tau_y) at significance (\varepsilon) so in-distribution points satisfy

![Threshold](../../../assets/thres2.svg)

Empirically, (\tau_y) is the ((1-\varepsilon))-quantile of calibration scores in class (y).

---

## 🛠️ Evaluator Variants & Asymptotics

We follow the paper’s notation: dataset size (n), proper-training fraction (p), calibration fraction (1-p), and (k = 1/(1-p)) folds. Assuming per-class calibration scores are pre-sorted or histogrammed, test-time lookup is ( \mathcal{O}(1) ).

| Evaluator             | Calibration Complexity                    | Test (per-sample)  | Notes                                                                                     |
| --------------------- | ----------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------- |
| **TCE**               | ( \mathcal{O}(n^2) )                      | ( \mathcal{O}(1) ) | Per-sample refits (impractical for streaming).                                            |
| **Approx-TCE**        | ( \mathcal{O}!\big(\tfrac{n}{1-p}\big) )  | ( \mathcal{O}(1) ) | Caches predictions over fixed folds; avoids per-sample refits.                            |
| **ICE**               | ( \mathcal{O}(pn) )                       | ( \mathcal{O}(1) ) | One fit on proper set; one held-out calibration bag.                                      |
| **CCE**               | ( \mathcal{O}!\big(\tfrac{pn}{1-p}\big) ) | ( \mathcal{O}(1) ) | (k) refits across folds; aggregate scores/p-values.                                       |
| **Approx-CCE (ours)** | ( \mathcal{O}(pn) )                       | ( \mathcal{O}(1) ) | Train once on proper set; compute scores across (k) calibration folds without (k) refits. |

**Intuition per variant (matching the paper):**

* **CCE.** (k)-fold proper training + calibration; aggregates fold scores to stabilize p-values. Cost (\mathcal{O}!\big(\frac{pn}{1-p}\big)).
* **ICE.** Single proper-set fit, one calibration bag; (\mathcal{O}(pn)).
* **Approx-TCE.** Streamlines Transductive CE by caching predictions over fixed folds; calibration ( \mathcal{O}!\big(\tfrac{n}{1-p}\big) ).
* **Approx-CCE (proposed).** **Train once** on (\approx pn) samples, then linearly score all (n) and compute per-class quantiles; calibration dominated by the single fit ⇒ ( \mathcal{O}(pn) ). Test-time remains ( \mathcal{O}(1) ).

---

## 🌟 Approx-CCE (Proposed)

**Motivation.** CCE is robust but expensive in streaming due to (k) refits.
**Design.** Train a **single shared model**; partition calibration into (k) folds to collect CCE-like coverage **without refitting**; thresholds are per-class ((1-\alpha))-quantiles.
**Advantages.**

* **Efficiency:** ICE-like calibration cost ((\mathcal{O}(pn))) with CCE-style score coverage.
* **Online readiness:** Parallelizable score computation; constant-time p-value lookup.
* **Compatibility:** Works with any probabilistic classifier exposing `fit/predict_proba`.

---

## ⚙️ Usage

```python
from core.conformalEval.conformal_evaluators import ConformalEvaluator

# Instantiate (5% significance)
ce = ConformalEvaluator(
    evaluator_type='approx_cce',  # 'ice' | 'cce' | 'tce' | 'approx_tce' | 'approx_cce'
    model=my_model,               # implements fit(), predict(), predict_proba()
    significance=0.05,
    calibration_split=0.2,        # used for ICE/Approx-CCE proper/cal splits when applicable
    folds=5                       # used for CCE/Approx-TCE/Approx-CCE scoring partitions
)

# Calibrate on historical data
ce.calibrate(X_cal, y_cal)

# Score new data
p_vals      = ce.predict_p_values(X_new)         # shape [n_samples]
y_pred      = ce.evaluator.predict(X_new)
drift_mask  = p_vals < ce.thresholds_[y_pred]    # per-class thresholds

if drift_mask.any():
    print("⚠️ Concept drift detected — consider retraining/recalibration.")
```

---

## 📚 Formulas (as used in the paper)

1. **Nonconformity:**
   ![Formula](../../../assets/form1.svg)
2. **p-value:**
   ![Formula](../../../assets/form2.svg)
3. **Threshold / Quantile rule:**
   ![Formula](../../../assets/form3.svg)

---

> **Model interface required**
>
> * `fit(X, y)`
> * `predict(X)`
> * `predict_proba(X)`  *(class-probabilities needed for many NCMs)*

---

For implementation details, see `utils.py` (calibration buffers, empirical CDF/histogram lookups).

--- 

## 📢 Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)
