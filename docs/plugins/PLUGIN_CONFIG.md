## Plugin Config Contract (v1)

### Purpose

ml-harness separates **orchestration** (core) from **model logic** (plugins).
To keep runs reproducible and sweeps ergonomic, every plugin follows a consistent convention for configuration (“hyperparameters”).

This contract ensures:

- all tweakable knobs live in one place
- run configuration is human-readable (YAML)
- sweeps can override values predictably
- configs are validated before execution
- tools/agents (e.g. Codex) have unambiguous rules about where to put parameters

---

## Key concepts

### Envelope vs payload

A run is composed of:

- **Run envelope**: stable orchestration fields owned by core
  (plugin key, dataset id, tags, seed, etc.)
- **Plugin payload**: plugin-owned configuration
  (model hyperparameters, training settings, evaluation settings, preprocessing settings)

Core treats plugin payload as opaque data _except_ for loading defaults, applying overrides, and validating via the plugin’s model.

---

## Run YAML format

A single run should be configurable with a single YAML file.

Recommended top-level keys:

- `plugin`: plugin identity
- `run`: core-owned execution metadata (tags, seed, strict mode)
- `data`: dataset binding inputs
- `params`: plugin hyperparameters (plugin-owned payload)

Example:

```yaml
plugin:
  key: hmm.fx_daily

run:
  experiment: mlh-hmm-fx
  tags:
    purpose: realdata
  seed: 42
  strict: true

data:
  dataset_id: fx:latent_returns_daily:v9
  dataset_path: ${HMM_DATASET_PATH}
  split_name: default

params:
  model:
    n_components: 3
    covariance_type: diag
    transmat_prior_strength: 50.0
    transmat_prior_mode: sticky_diag
  train:
    n_init: 5
    n_iter: 300
    tol: 1e-3
    init_strategy: kmeans
  preprocess:
    scaler: robust
    winsorize_vol: false
  eval:
    eval_scheme: last_n_days
    eval_last_n_days: 252
```

Notes:

- Environment variable substitution is allowed for values like `dataset_path`.
- Plugins should provide defaults so the YAML can be minimal (override-only).

---

## Plugin requirements

Each plugin SHOULD provide:

### 1) A typed params model

A dataclass or Pydantic model describing all supported hyperparameters, with defaults.

Goals:

- centralize knobs
- enable validation
- provide documentation via field names/types/constraints

### 2) Default params YAML (optional but recommended)

A human-friendly YAML file containing the default params structure.

This is used for:

- generating example configs
- composing defaults + overrides
- documentation

### 3) Validation entrypoint

The plugin must be able to validate a params dict before execution, producing friendly errors.

---

## Strict mode

`strict` controls how unknown keys are handled in `params`.

### strict: true (default)

- unknown keys in `params` are an error
- wrong types are an error
- missing required keys are an error

### strict: false

- unknown keys are allowed and preserved as `extras`
- known keys are validated
- plugins may optionally consume `extras`

Strict mode is designed to:

- keep configs reliable by default
- allow flexibility for experimental plugins

---

## Sweep configuration

Sweeps are provided as a second YAML file that only describes overrides.

Sweeps should target dot-path keys within the run config, typically under `params.*`.

Example:

```yaml
sweep:
  mode: grid
  overrides:
    params.model.n_components: [2, 3, 4, 5]
    params.train.n_iter: [200, 300]
```

Core will:

- load base run YAML
- expand sweep into multiple variants
- apply overrides to produce resolved configs
- validate each resolved config before running

---

## Output requirements

Core will always write (and upload) the resolved config for each run as an artifact, e.g.:

- `resolved/run_config.yaml` (or json)
- `summary/run_summary.json`

This guarantees every run is reproducible from artifacts alone.

---

## Authoring rules

For plugin authors (and automation tools like Codex):

- **Never** hardcode hyperparameters inside model/training code.
- All tweakable values must live in:
  - the plugin’s params model (and defaults)
  - the run YAML overrides (or sweep overrides)

This keeps experimentation transparent and sweepable.
