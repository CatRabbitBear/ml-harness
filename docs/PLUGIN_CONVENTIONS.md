# `PLUGIN_CONVENTIONS.md`

## Purpose

Plugins are intentionally **flexible and lightweight**. Core provides orchestration and stable services; plugins decide how to train, evaluate, and log. This document defines **repeatable conventions** for plugin structure, logging, artifacts, and dependency management so plugins don’t become inconsistent snowflakes as the repo grows.

---

## Core boundary

### What core guarantees to plugins

Plugins run via the stable `Plugin` contract and receive a `RunContext` containing:

- `context.run_id`: stable id for the run (created by core)
- `context.spec`: the `RunSpec` that requested the run
- `context.tracking`: a `TrackingClient` facade (MLflow-backed or fake)
- `context.artifact_dir`: a writable per-run directory (`<CORE_ARTIFACT_ROOT>/runs/<run_id>`)
- `context.logger`: a logger scoped to the run

Core owns `tracking.start_run(...)` and `tracking.end_run(...)`. Plugins must not manage run lifecycle.

### What plugins own

- The training/eval lifecycle within a run
- When and how to call `context.tracking.log_*` during training
- What artifacts are written and logged
- Any model/framework dependencies (kept out of core)

---

## Plugin package layout

Each plugin should be a small package under `plugins/<plugin_name>/` so it can grow cleanly.

Recommended structure:

- `plugins/<plugin_name>/`
  - `__init__.py` (export the plugin class)
  - `plugin.py` (implements `Plugin`; keeps `run()` thin)
  - `config.py` (parse/validate `RunSpec.data_spec`)
  - `data.py` (data creation/loading + train/val/test split)
  - `train.py` (fit/predict loop)
  - `eval.py` (metrics + plots)
  - `artifacts.py` (paths + write helpers)
  - `README.md` (what it does + what it logs + example `RunSpec`)
  - `tests/` (plugin-level tests, minimal)

This layout is a convention, not a hard rule. The goal is that `plugin.py` stays readable and “glue-like”, and complexity lives in helpers.

---

## `run()` shape

`Plugin.run(spec, *, context)` should read as orchestration glue:

- parse config from `spec.data_spec`
- seed randomness using `spec.seed` (if provided)
- build/load data
- train
- evaluate
- write artifacts under `context.artifact_dir`
- log params/metrics/artifacts using `context.tracking`
- return a `RunResult` with small, useful `outputs`

`run()` should not contain the full training implementation inline.

---

## Artifact conventions

Plugins must treat `context.artifact_dir` as their only guaranteed local workspace.

Standard subfolders (recommended):

- `data/` — summaries and metadata (avoid logging full datasets by default)
- `models/` — model files (e.g., `model.joblib`, checkpoints)
- `metrics/` — metrics JSON and structured outputs
- `plots/` — figures (e.g., confusion matrix, ROC curve)
- `reports/` — optional markdown/html reports

Log to tracking using matching `artifact_path` values (`data`, `models`, `metrics`, `plots`, `reports`) so the MLflow UI is predictable.

Notes:

- Core also writes `summary/run_summary.json` and (best-effort) logs it.
- Core may write `errors/exception.txt` on plugin failure and (best-effort) log it.
- Plugins should not assume core’s summary/error layout beyond those filenames existing.

---

## Tracking conventions

Plugins log via `context.tracking` (the `TrackingClient` facade). Plugins may log during training at any cadence they choose.

### Parameters

Use structured keys so runs remain searchable:

- `data.*` for dataset or synthetic generation config
- `model.*` for hyperparameters and model identity
- `split.*` for train/val/test ratios and split policy
- `seed` as a param or tag (if meaningful for the plugin)

Examples:

- `data.n_samples`, `data.n_features`
- `model.type`, `model.max_depth`
- `split.train`, `split.val`, `split.test`

### Metrics

Use explicit names that include split and (if needed) averaging:

- `train_accuracy`, `val_accuracy`, `test_accuracy`
- `train_f1_macro`, `val_f1_macro`, `test_f1_macro`

For iterative training, use `step` consistently (epoch, iteration, etc.):

- `context.tracking.log_metric("train_loss", loss, step=epoch)`

Avoid exploding the metric namespace. Prefer a single JSON artifact for large structured results.

### Tags

Use tags for coarse classification (environment, purpose, stage) that help filtering in MLflow:

- `purpose=smoke|synthetic|realdata`
- `plugin=<plugin_key>` (optional redundancy)
- `pipeline=train|score` (if used)

---

## Dependency management (Milestone 3.0)

Goal: prevent “dependency blow-up” while keeping plugin development ergonomic.

### Rule: core dependencies stay minimal

Core must not require heavy ML deps (e.g., sklearn/torch/transformers). Plugins may depend on them, but only behind optional installs.

### Option A: repo-level extras

The repo maintains a single `pyproject.toml`. Plugin dependencies are declared as **optional extras** and are not installed by default.

Conventions:

- One extra per plugin family or major stack, e.g.:
  - `sklearn` (numpy, scikit-learn, joblib, matplotlib)
  - `torch` (torch, torchvision, etc.)
  - `nlp` (transformers, tokenizers, datasets, etc.)

- Extras should be additive and composable.

Developer usage examples:

- Core-only dev: `pip install -e .`
- Sklearn plugin dev: `pip install -e ".[sklearn]"`
- Multiple stacks: `pip install -e ".[sklearn,nlp]"`

CI should include at least:

- core-only test job (ensures minimal deps remain sufficient)
- one or more “extras” test jobs (ensures plugin deps resolve and run)

### Option B: lazy imports inside plugins

Even when extras exist, plugins should avoid heavy imports at module import time.

Conventions:

- Import heavy libraries inside `run()` or a helper used by `run()`
- If a dependency is missing, fail with a clear message indicating which extra to install

Rationale:

- keeps importing `core` and listing plugins cheap
- prevents failures in environments that install only core
- produces clearer error messages than an import crash at module import time

---

## Testing conventions

- Core tests should not import plugin-only deps.
- Plugin tests can live under the plugin package (`plugins/<name>/tests/`) and may require extras.
- Prefer lightweight unit tests around:
  - config parsing
  - artifact writing paths
  - metric key naming
  - smoke-level training run with small data

---

## Non-goals of these conventions

- Forcing a single training loop style across all model types
- Enforcing fixed metrics or artifact sets for all plugins
- Turning plugins into a framework within a framework

The goal is consistency where it helps (layout, names, artifacts, dependency hygiene) and flexibility where it matters (model/training approach).

---

## Minimal checklist for a new plugin

- [ ] Has its own `plugins/<name>/` package
- [ ] Implements `Plugin` and returns stable `PluginInfo.key`
- [ ] `run()` is thin orchestration glue
- [ ] Writes outputs under `context.artifact_dir` using standard subfolders
- [ ] Logs params/metrics/artifacts via `context.tracking`
- [ ] Uses structured param/metric naming
- [ ] Plugin dependencies are behind a repo extra (not in core deps)
- [ ] Heavy imports are lazy (inside `run()`/helpers)
- [ ] Includes a short plugin `README.md` with an example `RunSpec`
