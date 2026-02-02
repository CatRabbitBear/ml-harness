Below is a **clean, commit-ready `docs/DATASETS.md`** that:

- keeps scope intentionally small (v0)
- makes the “hard edge” explicit
- avoids framework adapters
- avoids data-science pipeline creep
- is clear enough that Codex (and future-you) won’t drift it

You can drop this straight into the repo.

---

# Dataset Support (v0)

## Purpose

This document defines the **explicit dataset boundary** for `ml-harness`.

`ml-harness` is responsible for orchestrating, tracking, and promoting **models trained on prepared datasets**.
It is _not_ responsible for exploratory data analysis, feature engineering pipelines, or raw data ingestion.

The goal is to keep the system:

- reproducible
- auditable
- promotion-focused
- stable under iteration

---

## Core principle

> **Datasets consumed by `ml-harness` are already prepared and model-ready.**

Preparation (cleaning, feature engineering, experimentation) happens **upstream** and may be messy.
`ml-harness` begins at the point where data structure and semantics are stable.

---

## Supported dataset formats (v0)

### 1. Parquet (tabular) — **primary format**

Parquet is the canonical dataset format for v0.

**Rationale**

- efficient and columnar
- preserves schema and dtypes
- widely supported
- suitable for large datasets
- works naturally with pandas / pyarrow

**Assumptions**

- one row = one sample
- columns represent features and (optionally) target
- dataset schema is stable

---

## Dataset identity

Datasets are referenced by a logical identifier:

- `dataset_id`: a string (e.g. `local:iris_v1`, `parquet:customer_churn_2025q1`)

`dataset_id`:

- identifies _what_ dataset was used
- does **not** imply how it was created
- does **not** imply a storage backend or pipeline

---

## Required dataset metadata (v0)

For each run, the dataset loader must be able to supply or derive:

- `dataset_id`
- format (e.g. `parquet`)
- number of rows
- feature column names
- target column name (if supervised)
- basic label statistics (for classification)
- split policy or split manifest

This metadata must be logged as part of the run (e.g. via a JSON artifact).

---

## Train / validation / test splits

Splits must be **explicit and reproducible**.

Supported approaches (v0):

- **Predefined splits**
  - separate files
  - split column (e.g. `split ∈ {train,val,test}`)

- **Deterministic split policy**
  - ratios + seed
  - applied at load time

The split strategy must be logged.

---

## Canonical in-memory representations

`ml-harness` dataset loaders produce **framework-agnostic** data structures.

Recommended canonical forms:

- features: `pandas.DataFrame`
- targets: `pandas.Series`
- or equivalent tabular structures

Plugins are responsible for adapting these to:

- NumPy arrays
- PyTorch datasets
- TensorFlow datasets
- or any other framework-specific format

---

## Explicit non-goals (v0)

The following are intentionally **out of scope**:

- dataset downloading or ingestion
- exploratory cleaning or feature engineering
- schema inference or repair
- dataset versioning pipelines
- framework-specific adapters (e.g. parquet → torch Dataset)
- streaming / online datasets

These may be addressed by:

- upstream tools
- plugin-specific logic
- or future standalone projects

---

## Design rationale

This boundary exists to ensure that:

- experimentation remains flexible upstream
- the harness remains stable and predictable
- promotion-ready models can be reasoned about
- dataset assumptions are explicit and logged

`ml-harness` optimises for **clarity at the point of commitment**, not flexibility at the point of exploration.

---

## Future extensions (explicitly deferred)

Possible future additions (not commitments):

- additional formats (e.g. `npz`)
- dataset manifests
- content hashing
- integration with external dataset stores
- reusable dataset loaders

These will be driven by real usage pressure, not speculation.
