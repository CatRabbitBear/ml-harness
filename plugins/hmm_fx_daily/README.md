# HMM FX Daily (stub)

This plugin wires up the dataset + metadata loading pipeline for the
`latent_returns_daily` registry artifacts using `mlh-data`.

Current status:
- Loads the dataset from `HMM_DATASET_PATH` (or `data.dataset_path`)
- Fits a `hmmlearn.GaussianHMM` with configurable hyperparameters
- Logs metrics, tables, plots, and state time series to MLflow
- Writes model + scaler artifacts under `models/`

## Dependencies

Install with:

```bash
pip install -e ".[mlh-data,data]"
```

## Example RunSpec

```python
from core.contracts import RunSpec

spec = RunSpec(
    plugin_key="hmm.fx_daily",
    dataset_id="fx:latent_returns_daily:v6",
    data_spec={
        "data": {
            "dataset_path": r"D:\\fx-registry\\datasets\\latent_returns_daily\\v6",
            "split_name": "default",
        },
        "model": {
            "n_components": 4,
            "transmat_prior_strength": 20.0,
            "transmat_prior_mode": "sticky_diag",
        },
        "train": {
            "n_init": 5,
            "n_iter": 300,
            "tol": 1e-3,
            "init_strategy": "kmeans",
        },
        "preprocess": {
            "scaler": "standard",
            "winsorize_vol": False,
        },
        "eval": {
            "eval_scheme": "last_n_days",
            "eval_last_n_days": 252,
        },
    },
)
```

## Outputs

- `data/dataset_summary.json` (manifest/schema/roles/stats + split sizes)
- `data/state_timeseries.parquet`
- `data/state_summary.csv`
- `data/transmat.csv`
- `data/transmat_prior.csv`
- `plots/*.png`
- `models/hmm_model.joblib`
