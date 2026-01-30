# TODO: Artifact layout duplication (core workspace vs MLflow artifact store)

## Context
Milestone 2 proves end-to-end MLflow tracking works with a shared artifact root volume.
Currently, two directory trees appear under the same shared root:

- Core workspace (write-first):
  - `<CORE_ARTIFACT_ROOT>/runs/<run_id>/...`
- MLflow artifact store (log-to, MLflow managed):
  - `<MLFLOW_DEFAULT_ARTIFACT_ROOT>/<experiment_id>/<run_id>/artifacts/...`

This can look like “duplicate” data because MLflow copies logged artifacts from the core workspace into its own artifact store layout.

This is not a bug: it reflects two different roles:
- workspace for plugins/core to write files
- MLflow-managed store for logged artifacts visible in the UI

However, the current layout can be confusing and may grow storage usage.

## Why we are not changing it yet
- Requirements will likely evolve once real models produce larger artifacts.
- We have not fully nailed environment management (dev vs docker vs CI).
- Current behaviour is correct and stable; changing it now risks churn.

## Options

### Option A — Keep workspace + MLflow store (current)
Keep core writing into a per-run workspace directory and explicitly log selected artifacts to MLflow.

Pros:
- Backend-agnostic: plugins always write to a local Path.
- Clear semantics: "write → (optionally) log".
- Minimal coupling to MLflow.

Cons:
- Can look like duplication (workspace + logged copies).
- Storage can grow if we log large directories.

Potential tweak (non-breaking):
- Separate roots to make the roles obvious:
  - core workspace: `/shared/artifacts/work/runs/<run_id>/...`
  - mlflow store: `/shared/artifacts/mlruns/<experiment_id>/<run_id>/artifacts/...`
  - Set:
    - `CORE_ARTIFACT_ROOT=/shared/artifacts/work`
    - `mlflow --default-artifact-root /shared/artifacts/mlruns`

### Option C — Workspace is scratch + cleanup
Keep Option A’s semantics but delete the core workspace directory on success.
Keep workspace on failure for debugging.

Pros:
- Prevents storage growth while keeping backend-agnostic workflow.
- Retains failure artefacts locally.

Cons:
- Loses the local workspace after successful runs (unless explicitly retained).
- Requires clear policy (when to delete, what to retain).

## Decision (for now)
Defer changes. Continue with Option A (current) until:
- we start logging real model artifacts and observe size/usage
- env/config strategy is settled (docker vs local vs CI)
- we decide whether the workspace is meant as a permanent cache or temporary scratch

## Trigger to revisit
Revisit this decision when we add a real training plugin that logs:
- a model artifact (potentially large)
- evaluation plots/reports
- dataset snapshots / feature stats

At that point, choose between:
- Option A with separated roots (work vs mlruns)
- Option C (cleanup on success)
