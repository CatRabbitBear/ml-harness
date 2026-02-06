# Milestone 2 — End-to-End MLflow Tracking (Local, Dockerised)

## Purpose
Milestone 2 proves that the core orchestration engine can run against a **real MLflow tracking server** and produce runs + artifacts that are visible in the MLflow UI.

No real models, datasets, or promotions are required. This milestone is about infrastructure correctness and lifecycle integration, not ML quality.

---

## High-level goals

1) Stand up a local MLflow tracking stack using Docker Compose
2) Ensure artifact storage works correctly via a shared filesystem
3) Demonstrate an end-to-end run from the app into MLflow UI

Completion means: *a run launched by the app container is visible in the MLflow UI with metrics and artifacts.*

---

## Target architecture (local)

Services:
- **app**
  Runs the orchestration engine and calls `run_pipeline(...)`.

- **mlflow**
  MLflow Tracking Server (HTTP API).

- **postgres**
  Backend store for MLflow metadata.

---

## Artifact storage (canonical)
- A single shared volume is mounted into:
  - `app` at `/shared/artifacts`
  - `mlflow` at `/shared/artifacts`
- Environment variable:
  - `CORE_ARTIFACT_ROOT=/shared/artifacts`
- MLflow server is configured with:
  - `--default-artifact-root /shared/artifacts`

Artifacts written by the app must be readable by the MLflow server without copying.

---

## MLflow configuration (v1 constraints)
- Backend store: Postgres
- Artifact store: local filesystem (shared volume)
- No MinIO / S3 / remote object store
- No auth, no TLS, no multi-tenant concerns
- Single experiment is sufficient (name can be fixed or env-driven)

---

## Execution model
- App container points at MLflow server via:
  - `MLFLOW_TRACKING_URI=http://mlflow:5000` (or equivalent config)
- App builds:
  - a plugin registry
  - a TrackingClient bound to the MLflow server
- App invokes `run_pipeline(...)` exactly once (smoke run)

---

## Success criteria
Milestone 2 is complete when:
- `docker compose up` starts all services successfully
- MLflow UI loads in a browser
- A run appears in the UI after the app executes
- The run shows:
  - at least one metric
  - at least one artifact written under the shared artifact root
- Artifact paths on disk match MLflow’s recorded locations

---

## Non-goals (explicit)
- No real ML training or datasets
- No promotions or model registry workflows
- No Kubernetes
- No remote artifact stores
- No production hardening (auth, secrets, scaling)

---

## Guiding principles for tasks
- Prefer correctness over completeness
- Keep tasks small and composable
- Do not add new abstractions unless required
- If something feels “future-proof”, leave a TODO instead

This milestone exists to validate the architecture, not to expand it.
