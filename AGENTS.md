# AGENTS.md
## ml-harness – Agent Instructions

This repository is structured around a **clean, isolated core engine** with
injected concrete implementations (plugins, apps). Agents MUST follow the
architectural rules below when making changes.

---

## 1. Project intent (read first)

`ml-harness` is a framework for orchestrating ML workflows (dataset building,
training, evaluation, promotion) with strong emphasis on:

- clean separation of concerns
- reproducibility
- testability
- minimal coupling between core logic and concrete ML implementations

The core package is designed to be **potentially extractable as a standalone
library** in the future.

see STYLEGUIDE.md for code style guidance

---

## Target runtime architecture (3-container stack)

### Overview

The project is designed to run as a small, production-shaped stack:

1. **app**
   Hosts the orchestration runtime (“MLflow++ core”) and the concrete plugins.
   Responsibilities:

   * resolve plugin from injected registry
   * start/finish MLflow runs
   * create a per-run working directory
   * call plugin code with a typed `RunContext`
   * log metrics/params/artifacts/models via the tracking façade

2. **mlflow**
   MLflow Tracking Server (HTTP API).
   Responsibilities:

   * store experiment/run metadata in Postgres
   * serve the tracking API used by `app`
   * reference artifact files on the shared artifact volume (artifact root)

3. **postgres**
   MLflow backend store.
   Responsibilities:

   * persist MLflow experiments, runs, params, metrics, tags, model registry info

### Shared artifact volume (critical)

`app` and `mlflow` must share the **same mounted volume** for artifacts.

* A named Docker volume is mounted into both containers at the same path, e.g.:

  * `app`: `/shared/artifacts`
  * `mlflow`: `/shared/artifacts`

* MLflow is configured with:

  * `--default-artifact-root /shared/artifacts`

* Core creates a per-run directory under the shared root, e.g.:

  * `/shared/artifacts/runs/<run_id>/`

Plugins may write artifacts to this directory and log them via `context.tracking.log_artifact(...)` (preferred “log as you go”), or core can log the directory at the end.

### Networking expectations

* `app` talks to `mlflow` over HTTP (tracking URI points at the mlflow container).
* `mlflow` talks to `postgres` via the internal Docker network.
* `postgres` is not exposed publicly in typical deployments.

### Data flow (training run)

1. `app` receives a `RunSpec`
2. `app` starts an MLflow run (tags + run name)
3. `app` creates `/shared/artifacts/runs/<run_id>/`
4. `app` calls `plugin.run(spec, context=RunContext(...))`
5. plugin logs metrics/params/artifacts/models via `context.tracking`
6. `app` finalizes the run (status, summary artifact, end_run)

### Why this shape

* Core owns run lifecycle for consistency (tags, structure, failure handling).
* Plugins stay MLflow-agnostic (depend only on `RunContext` + tracking façade).
* Artifacts are accessible to MLflow server via the shared volume.
* Promotions/model registry steps can be added later by consuming MLflow-tracked outputs (no need to rerun training).

---

## 2. Repository structure (authoritative)

High-level layout:

ml-harness/
core/ # Pure orchestration engine + contracts
plugins/ # Concrete ML implementations (HMM, RL, etc.)
apps/ # API / CLI / services that wire everything together
deploy/ # Docker / infra wiring
docs/ # Design and architecture docs


### Core package layout (IMPORTANT)

core/
src/
core/
api.py
contracts/
orchestration/
mlflow/
testing/ # INTERNAL test helpers (not public API)


Key rules:
- `core/src/core` is the **only** importable `core` package.
- The outer `core/` and `core/src/` directories are NOT Python packages.
- All imports use `from core.<module> import ...` (never `core.src.core...`).

---

## 3. Dependency direction (DO NOT VIOLATE)

Allowed dependencies:

plugins -> core.contracts
apps -> core.api / core.contracts
core -> (stdlib only, plus minimal infra libs)


Forbidden dependencies:
- `core` importing from `plugins`
- `core` importing from `apps`
- contracts importing the contracts *package* (`from core.contracts import ...`)
  inside submodules (causes circular imports)

Core must remain usable without any plugins present.

---

## 4. Plugin architecture (authoritative)

- Core defines **interfaces only** (`Plugin`, `PluginRegistry`)
- Concrete plugins live outside core
- Plugins are injected at runtime via a registry

Public entrypoint:

```python
run_pipeline(spec: RunSpec, *, registry: PluginRegistry) -> RunResult
```

Core does NOT discover plugins itself.
5. Testing conventions

    Tests live under core/tests/, plugins/**/tests, apps/**/tests

    Shared test helpers live in core.testing.*

    Test helpers are allowed to be packaged; tests themselves are not public API

    Prefer pytest.raises over manual try/except

    Avoid import hacks, sys.path mutation, or cwd-dependent behaviour

6. Agent operating rules

When modifying code, agents MUST:

    Respect existing contracts and public APIs

    Avoid introducing new dependencies into core without explicit instruction

    Prefer small, composable changes

    Leave clear TODOs rather than speculative abstractions

    Flag architectural uncertainty instead of guessing

If a change would:

    blur core/plugin boundaries

    require circular imports

    introduce implicit global state

STOP and ask for clarification.
7. Style & hygiene

    All public functions/classes require docstrings

    Core APIs should be boring, explicit, and stable

    Avoid “magic” behaviour (implicit discovery, side effects on import)

    Optimise for clarity over cleverness
