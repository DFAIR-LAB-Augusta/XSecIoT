# Recommended Labels

Use these labels for consistent triage and planning.

## Type
- `type: bug` — Something is broken
- `type: feature` — New behavior/capability
- `type: refactor` — Code cleanup without behavior change
- `type: docs` — Documentation updates
- `type: tests` — Test additions/changes
- `type: chore` — Maintenance, deps, tooling

## Priority
- `priority: p0` — Drop everything / blocking
- `priority: p1` — High
- `priority: p2` — Normal
- `priority: p3` — Low

## Area (FIRCE)
- `area: simulation` — `ce_simulation.py`, streaming loop
- `area: training` — `ce_model_training.py`, pipelines, artifacts
- `area: conformal-eval` — `src/core/conformalEval/*`
- `area: models` — `src/core/models/*` (FFN/MLP/etc)
- `area: preprocessing` — column/schema handling, cleaning
- `area: logging` — rolling CSV, circular logger, plots
- `area: datasets` — DFAIR/UNSW integration, labeling scripts
- `area: ci` — GitHub Actions, checks
- `area: docs` — README, usage, paper artifacts

## Status
- `status: needs-triage`
- `status: blocked`
- `status: ready`
- `status: in-progress`
- `status: needs-review`

## Difficulty / Help Wanted
- `good first issue`
- `help wanted`
