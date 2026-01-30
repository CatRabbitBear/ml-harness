## Artifact storage

This project uses a shared artifact directory so that training runs and the MLflow tracking server can see the same files.

### Environment variable
Set a single environment variable to control where artifacts are written:

```bash
export CORE_ARTIFACT_ROOT=/shared/artifacts
```

### Docker usage (recommended)

In Docker deployments, this path should be a mounted volume that is shared between:

* the application container (core + plugins)
* the MLflow tracking server container

MLflow should be configured with the same path as its default artifact root.

### Local development

No configuration is required for local development.

If `CORE_ARTIFACT_ROOT` is not set, core will automatically fall back to:

* a local `.artifacts/` directory in the project root, or
* a temporary directory if needed

This ensures runs are “forgettable” and experimentation never fails due to missing configuration

## MLflow local stack (Docker Compose)

From the repo root, start the local MLflow + Postgres stack:

```bash
docker compose -f deploy/docker-compose.yml up
```

Then open the MLflow UI at:

```text
http://localhost:5000
```

Artifacts are stored on a shared Docker volume mounted at `/shared/artifacts` in the MLflow container.
