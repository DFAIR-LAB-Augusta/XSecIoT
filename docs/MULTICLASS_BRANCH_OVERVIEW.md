# `multiclass` Branch: What Changed and Why

**Purpose of this document:** a single entry point for a future session (human or agent) picking up work on this branch, without needing to replay the full history of 30 merged PRs. Read this first; drill into the linked plan docs (`docs/superpowers/plans/`) or PRs for implementation detail on any specific piece.

Branch `multiclass` diverged from `main` at `aa353f3` and has since merged 30 PRs (#102–#147, working PR numbers — the underlying GitHub issue numbers referenced throughout this doc are different and lower, e.g. issue #85 was implemented in PR #103). As of the last update to this document, every merged PR's CI was green and every issue closed below was confirmed done by the person merging, not assumed.

## The short version

This branch took the repo from **binary-only** classification to **binary + multiclass**, then used that as a foundation for a broad test-coverage push, a cluster of real bug fixes, and — as the current frontier — an entirely new subsystem (`src/firce/novelty/`) implementing the dissertation's **Direction 3**: detecting *emerging/unknown* attack behavior (not just classifying among known classes), explaining why an event was flagged, generating a human-readable report via a local LLM, and evaluating the whole pipeline end-to-end.

## Work streams, in the order they happened

### 1. Multiclass support roadmap (issues #85–#95, PRs #102–#113)

Extended the existing binary-only FIRCE pipeline (train → calibrate CE monitor → simulate → detect drift → retrain) to also support multiclass classification, end to end:

- `src/firce/models/mlp_ce_base.py` / `mlp_ce_multiclass.py`, `feedforward_base.py` / `feedforward_multiclass.py` — split the previously binary-only model classes into shared base + binary/multiclass subclasses.
- `src/firce/ce_model_training.py::train_ce_multiclass` — new multiclass training entrypoint mirroring `train_ce_binary`, across every model variant (dt/knn/rf/svm/xgb/feedforward).
- `src/firce/runtime/bootstrap.py`, `retraining.py`, `inference.py` — wired multiclass through model-artifact bootstrapping, drift-triggered retraining, and live prediction/correctness-tracking.
- `src/firce/utils/config.py` — `SimulationConfig` validation for multiclass mode; a default rolling-schema gap for `MC_Label` was found and fixed here (issue #92).
- `scripts/mc_labeling.py` — CLI for deriving multiclass labels onto a dataset that only has binary labels.
- Multiclass coverage added across the CE evaluators (`ICE`/`CCE`/`ApproxTCE`/`ApproxCCE`) and a CADE-monitor compatibility spike (fixed a label-encoding bug in the monitor-fit path along the way).

**Read next:** `docs/superpowers/plans/2026-09-0{5,7,8}-mc-*.md` for the individual plan docs; `project_xseciot_multiclass_roadmap.md` and the `project_xseciot_mc_*_plan.md` files in the memory system for a session-by-session account.

### 2. General test-coverage push (issue #104, PRs #115–#119) + bugs found along the way

Issue #104 was filed after the roadmap work exposed that most of `src/firce/`'s runtime files had little or no active test coverage (many existing test files were stub smoke tests or fully commented out). Four sequential PRs built a real test pyramid: shared `conftest.py` fixtures, binary-path component tests mirroring the multiclass ones, CE-evaluator/drift-monitor tests, and a real end-to-end simulation test using sampled real flow data (`tests/fixtures/ce_flows_e2e_*.csv`).

Three real bugs were found and fixed as follow-up issues during this push, each independently scoped and merged:
- **#114** (PR #120): `train_ce_binary`'s label normalization silently broke under pandas 3's new string-dtype default (`dtype == object` no longer matched).
- **#108** (PR #122): the UNSW rolling-log schema had no slot for `MC_Label`, and `load_training_frame` never derived it for the UNSW+multiclass combination.
- **#121** (PR #123): `_prepare_chunk` hardcoded `clean_data(chunk, False)`, ignoring `config.is_unsw` — silently dropped every column for UNSW-format input.

### 3. FIRCE-focus audit and the session's most severe bug (PRs #124, #127–#133)

Mid-session, explicit user direction ("torch is the grind fuck keras", "firce should be where big multiclass focus is") triggered an audit: 10 stale pre-2026 `[FIRCE-MC]` issues referencing the old `src/core/` layout were closed as superseded, and deeper test coverage was added for the newer `firce/*` pipeline (issue #118, PRs #124 and #130–#133) while `fire/*` (the older, Keras-based batch pipeline) was deliberately deprioritized.

**The most severe bug found this entire branch** (issue #125, PR #127): `firce/runtime/bootstrap.py::_build_monitor_model`'s default case returned the live production model **by reference** instead of a clone. `ICE.calibrate()` refits its model in place as part of calibration — so every CE-monitor-backed simulation using a classical model variant was silently corrupting (binary) or crashing (multiclass, due to a label-encoding mismatch) the actual model being used for real predictions. This had existed since issue #92. Fixed by returning `clone_model(model)` — reusing the existing `conformalEval/utils.py` utility, not inventing a new one. PR #128 then hardened the regression test across DT/KNN/RF/SVM variants.

A related issue (#126, PR #129) found that `fire.models`'s legacy artifact directories (`binary_models/`, `multi_class_models/`) collided filename-for-filename with `firce.ce_model_training`'s own output directories — renamed to `*_legacy_fire` to namespace them apart.

**Read next:** `project_xseciot_mc_e2e_and_shared_model_bug_125.md` and `project_xseciot_paper_lineage.md` in the memory system for the full incident writeup and the fire→firce→fades→firce-mc research lineage that motivated the focus shift.

### 4. Direction 3: novelty detection, XAI, LLM reporting (issues #96–#101, PRs #134–#136, #143–#144)

This is the current frontier of the branch and the reason the new `src/firce/novelty/` package exists. Per the dissertation proposal (Section 4.4), Direction 3 extends the closed-world multiclass classifier to detect **emerging/unknown** behavior not present in the training label space, explain why, and generate a human-readable report:

- **`decision_rules.py`** (#97, PR #134) — `is_novel(probas, all_class_p_values, tau, alpha)`, the primary decision rule: `(max softmax < tau) AND (all per-class conformal p-values < alpha)`. Also `margin_confidence`, `entropy_confidence`, `temperature_scaled_confidence`, `conformal_prediction_set_size` as alternative criteria the proposal calls out for comparison. `compute_all_class_p_values` fills a real gap: the existing CE evaluators (`ICE`/`CCE`) only ever exposed a p-value for the *predicted* class, not all classes — needed for the `AND`-across-all-classes rule.
- **`explain.py`** (#98, PR #135) — `explain_with_shap`/`explain_with_lime` producing structured (not plot-file) attributions for a single flagged event, plus `select_events_to_explain` implementing three selective-generation policies (`unknown_only`/`sampled`/`windowed`) so explanation generation respects a latency budget instead of running on every row.
- **`llm_reporting.py`** (#99, PR #136; extended by #141/#138, see below) — a **local-only** (not hosted-API) LLM reporting pipeline, per explicit user direction. `TransformersLocalBackend` wraps any HF-format checkpoint; `build_report_prompt`/`parse_report_output`/`generate_report` form the original regex-based pipeline.
- **`mitre_mapping.py`** (#100, PR #143, optional) — heuristic TF-IDF similarity mapping from a generated report to MITRE ATT&CK techniques. The reference set (`data/mitre_technique_subset.json`) is deliberately scoped to the 14 techniques the user's separate CAPEX attack-generation repo already targets, not a generic list — fetched from the real MITRE ATT&CK STIX bundle and cross-checked, not typed from memory.
- **`evaluation.py`** (#101, PR #144) — `evaluate_closed_world`/`evaluate_open_world_novelty`/`evaluate_explanation_utility`, the three-stage harness the proposal calls for. Building a genuine open-world test (train on K-1 of K classes, evaluate on the held-out one) surfaced another real bug: all four CE evaluators' `precision_score(..., average='binary' if len(np.unique(y))==2 else 'weighted')` assumed any 2-class label set was conventional `{0,1}` — crashed on a 2-of-3-class string-labeled subset. Fixed with a shared `pick_average_strategy` utility.

**A recurring, load-bearing finding across this whole work stream:** `DecisionTreeClassifier` (used in most of this codebase's existing test fixtures) is too discrete/overconfident to demonstrate novelty detection meaningfully — it saturates to 0.0/1.0 confidence even on genuinely out-of-distribution input. `RandomForestClassifier` gives a real, usable confidence signal instead. Use RF, not DT, for any future novelty-detection test fixture or real deployment tuning in this codebase.

### 5. Wiring Direction 3 into the live runtime + follow-ups (issues #142, #141, #138 — all filed as "improve on #99's basic setup" — PRs #145, #146, #147)

After #96–#101 landed, #97/#98/#99 were real, tested, but **not called from anywhere** in the live simulation path. Three follow-up issues closed that gap and hardened it further:

- **#142** (PR #145) — `ConformalDriftMonitor` gained `.model`/`.calibration_scores` properties (reusing the exact calibration state drift detection already computes). New `_score_chunk_novelty`/`_generate_novelty_reports` in `runtime/inference.py`, wired into `process_chunk`. Nine new `SimulationConfig.novelty_*` fields gate everything, all defaulting to **off** — zero behavior change unless explicitly enabled. `SimulationRuntime` gained `.novelty_reports`/`.llm_backend`.
- **#141** (PR #146) — `generate_structured_report`, a grammar-constrained (via `outlines`) alternative to the original regex-based parsing, guaranteeing valid structured output by construction (via a bounded Pydantic schema — an *unbounded* schema field lets even an untrained model generate forever without closing a JSON string, confirmed by direct execution).
- **#138** (PR #147) — `build_report_prompt` gained a `strategy` parameter (`zero_shot`/`few_shot`/`cot`), default unchanged, plus `evaluate_prompt_strategies` to compare compliance rate across them.

**Still open, not part of this branch's work so far:**
- **#137** — a model-selection matrix across specific hardware tiers (workstation GPU / laptop GPU / Apple Silicon / commodity CPU / Raspberry Pi). Explicitly paused: it asks for real on-device measurements this sandbox environment cannot produce.
- **#139**, **#140** — knowledge distillation and supervised fine-tuning of a local reporting model. Explicitly the user's own future work, filed as scoping placeholders only.

## Key files to know about

| Path | What it is |
|---|---|
| `src/firce/novelty/decision_rules.py` | Novelty/unknown detection rule + alternatives |
| `src/firce/novelty/explain.py` | Structured SHAP/LIME explanations + selective-generation policy |
| `src/firce/novelty/llm_reporting.py` | Local LLM backend, prompt strategies, regex-based + grammar-constrained report generation |
| `src/firce/novelty/mitre_mapping.py` | Heuristic MITRE ATT&CK technique suggestion |
| `src/firce/novelty/evaluation.py` | Closed-world / open-world / explanation-utility evaluation harness |
| `src/firce/runtime/inference.py` | Live per-chunk prediction loop; `_score_chunk_novelty`/`_generate_novelty_reports` are the Direction 3 hook points |
| `src/firce/runtime/bootstrap.py` | Runtime construction (`initialize_simulation_runtime`); `_build_monitor_model`'s `clone_model` fix (issue #125) lives here |
| `src/firce/conformalEval/utils.py` | Shared CE utilities — `clone_model`, `pick_average_strategy`, per-class p-value/threshold helpers |
| `src/firce/utils/config.py` | `SimulationConfig` — every runtime knob, including the `novelty_*` fields |
| `src/fire/` | The older, Keras-based batch pipeline — deprioritized per explicit user direction, kept working but not a focus |
| `docs/superpowers/plans/` | One plan doc per issue/PR on this branch — the detailed "how," this doc is the "what and why" |

## Conventions this branch established (apply them if continuing this work)

- **Worktree-per-issue**: `git worktree add .worktrees/<branch> -b <branch> origin/multiclass`, work there, PR back into `multiclass` (not `dev`/`main`).
- **TDD with real execution, not mocks**: every bug in this document was found by writing a test that actually exercises the real code path first, then confirming the fix with a re-run — not guessed from reading code. LLM-related tests use a genuinely tiny (2-layer, 16-dim), randomly-initialized, fully offline-built model + tokenizer (see `tests/test_novelty_llm_reporting.py`) rather than mocking `.generate()`.
- **RandomForest over DecisionTree** for any test fixture or config that needs a real confidence signal (see Direction 3 section above).
- **`clone_model` before handing a trained model to anything that might refit it** (e.g. `ConformalDriftMonitor`) — issue #125's bug, and it was re-triggered once more (self-inflicted, in a test) while building #142's end-to-end test.
- Every PR ships with CI green (`pytest` + `ruff-style`) before being reported ready to merge; the person merging always merges manually — this repo's workflow never auto-merges.
