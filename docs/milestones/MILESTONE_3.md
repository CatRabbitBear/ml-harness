# Milestone 3 — Real Plugin, Real Stress

## Purpose

Milestone 3 introduces a **non-trivial “real” plugin** whose goal is to stress-test the orchestration, tracking, and artifact lifecycle under realistic conditions.

This milestone is **not** about production ML, optimisation, or performance.
It is about validating that:

- the current core contracts are sufficient for real work
- MLflow tracking is exercised in a meaningful way
- artifact handling is understandable and stable
- lessons about data handling emerge from practice, not speculation

---

## Scope overview

Milestone 3 is intentionally split into three sub-milestones:

- **3.0** Ensure alignment with PLUGIN_CONVENTIONS.md and .env use
- **3.1** Real model + synthetic data (runtime-generated)
- **3.2** Same model + real dataset (small, local)
- **3.3** Define what “good” looks like for a dataset pipeline (plan only)

Each stage builds on the previous one without introducing new infrastructure.

---

## 3.0 — Plugin groundwork & environment hygiene

### Goal

Prepare the repository for sustainable plugin development **before** introducing a real training plugin.

This step establishes:

* consistent plugin dependency management
* safe import behaviour for heavy ML libraries
* a simple, repeatable way to manage environment variables for local development

No new runtime features are introduced.

---

### Plugin dependency conventions

Before writing the first Milestone 3 plugin:

* Plugin dependencies must **not** be added to core requirements.
* Plugin dependencies must be declared as **optional extras** at the repository level.
* Core must remain installable and runnable without any plugin extras.

Each plugin:

* documents which extra(s) it requires
* lazily imports heavy dependencies inside `run()` or helper functions
* fails with a clear error message if required extras are not installed

These conventions are defined in `PLUGIN_CONVENTIONS.md` and are considered **mandatory** from Milestone 3 onward.

---

### Environment variable usage

Milestone 3 formalises the use of environment variables for:

* artifact root configuration (`CORE_ARTIFACT_ROOT`)
* MLflow configuration (`MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`)
* future credentials (e.g. API keys), without hardcoding them into code or specs

Environment variables are treated as **deployment / execution concerns**, not plugin configuration.

---

### Local development ergonomics

To support clean local workflows:

* A `.env` file may be used for local-only configuration
* `.env` files are not committed to version control
* Example files (e.g. `.env.example`) may be committed to document expected variables

Optionally:

* a small PowerShell helper script (e.g. `load-env.ps1`) may be added to load variables from `.env` into the current shell session

This script:

* is for developer convenience only
* does not affect core, plugins, or runtime behaviour
* is not required for CI or deployment

---

### Acceptance criteria (3.0)

* Plugin dependency rules are documented and agreed
* No plugin dependencies leak into core installs
* Core and tests run without plugin extras installed
* Environment variables are the single source of runtime configuration
* Local setup does not require manual export of many variables

---

### Non-goals for 3.0

* Automatic plugin discovery
* Per-plugin virtual environments
* Secrets management infrastructure
* Runtime loading of `.env` files inside core

---

## 3.1 — Real model + synthetic data

### Goal

Exercise the full run lifecycle using:

- a real ML model
- realistic metrics
- multiple artifacts (files + plots)
- synthetic data generated at runtime

This validates the system without any external data dependencies.

---

### Plugin characteristics

A new plugin is introduced under `plugins/`, implementing the existing `Plugin` interface.

The plugin:

- generates synthetic data at runtime
- performs train / validation / test splits
- trains a real model
- evaluates metrics
- logs metrics, params, and artifacts via `context.tracking`
- writes all intermediate files under `context.artifact_dir`

The plugin **must not**:

- manage MLflow runs directly
- assume any filesystem layout beyond `context.artifact_dir`
- depend on app-level or deployment-specific configuration

---

### Suggested experiment (non-binding)

A simple classification task is recommended for clarity and MLflow visibility:

- data generation via `sklearn.datasets.make_classification`
- model: `LogisticRegression` or `RandomForestClassifier`
- splits: train / validation / test (e.g. 70 / 15 / 15)

These are suggestions, not requirements; the important part is that the experiment is _real_ and multi-step.

---

### RunSpec usage

Configuration is carried via the existing `RunSpec` fields:

- `RunSpec.data_spec` contains synthetic data generation parameters
  - sample count, feature count, class separation, noise, etc.

- `RunSpec.seed` seeds all randomness (numpy / sklearn)
- `RunSpec.tags` may include experiment intent (e.g. `purpose=synthetic-test`)

No new fields or contracts are introduced.

---

### Metrics (examples)

Metrics should be logged with clear, explicit names, for example:

- `train_accuracy`
- `val_f1`
- `test_f1`
- `test_auc` (if applicable)

The exact metric set is flexible; clarity and consistency matter more than completeness.

---

### Artifacts (examples)

The plugin should produce and log multiple artifacts, such as:

- trained model file (e.g. `model.joblib`)
- `metrics.json`
- `data_summary.json` (shape, class balance, generation params)
- one or more plots (e.g. confusion matrix, ROC curve)

Artifacts are first written to `context.artifact_dir`, then logged via `context.tracking.log_artifact` or `log_artifacts`.

---

### Acceptance criteria (3.1)

- A single run produces:
  - meaningful params, metrics, and tags in MLflow
  - multiple visible artifacts in the MLflow UI

- The run completes successfully end-to-end
- Re-running with the same `RunSpec.seed` and `data_spec` yields reproducible structure and similar metrics
- No core contracts or APIs are changed

---

## 3.2 — Same plugin, real dataset

### Goal

Validate that the same plugin shape works with a real dataset, without introducing a dataset pipeline yet.

---

### Dataset choice

Use a **small, local, zero-dependency dataset**, such as:

- `sklearn.datasets.load_iris`
- `sklearn.datasets.load_breast_cancer`

The dataset must be available at runtime without downloads or credentials.

---

### Configuration approach

One of the following is sufficient:

- use `RunSpec.data_spec` to describe the dataset source
- or set `RunSpec.dataset_id` to a logical identifier (e.g. `sklearn:iris`)

No persistence or caching is required at this stage.

---

### Expectations

- The training / evaluation / logging flow remains unchanged
- Metrics and artifacts are logged in the same way as in 3.1
- Data summary artifacts reflect real dataset properties (feature count, label distribution)

---

### Acceptance criteria (3.2)

- The run behaves identically to 3.1 from an orchestration perspective
- MLflow contains comparable metrics and artifacts
- No special-case logic leaks into core or apps

---

## 3.3 — Define “good” for a dataset pipeline (plan only)

### Goal

Use the experience from 3.2 to define **requirements** for a future dataset pipeline, without implementing it yet.

This stage is documentation and decision-capture only.

---

### Topics to clarify

Based on practical experience, document:

- What `dataset_id` is expected to represent
  - immutable snapshot vs logical reference

- What dataset metadata must be logged for every run
  - schema, row counts, time ranges, feature summary

- What guarantees are required for reproducibility
  - deterministic builds, content hashes, seeds

- What responsibilities belong to:
  - core
  - plugins
  - a future dataset-builder component

---

### Output

A short design note (or update to an existing TODO doc) that:

- captures lessons learned from 3.1 / 3.2
- states constraints and invariants
- explicitly avoids committing to a concrete implementation

---

### Acceptance criteria (3.3)

- Dataset pipeline expectations are documented clearly
- No new code or abstractions are introduced
- Open questions are explicitly listed for a future milestone

---

## Non-goals for Milestone 3

- Production-grade data ingestion
- Dataset versioning infrastructure
- Scheduling or backfills
- Model registry or promotion logic
- Performance optimisation or scaling

---

## Exit condition

Milestone 3 is complete when:

- a realistic plugin exists and runs cleanly
- MLflow tracking has been exercised meaningfully
- artifact behaviour is well understood
- future dataset work is grounded in real usage, not guesswork
