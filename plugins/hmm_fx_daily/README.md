# HMM FX Daily (stub)

This plugin wires up the dataset + metadata loading pipeline for the
`latent_returns_daily` registry artifacts using `mlh-data`.

Current status:
- Loads the dataset from `HMM_DATASET_PATH` (or `data.dataset_path`)
- Applies the split spec from the dataset artifacts
- Writes `data/dataset_summary.json` into the run artifact dir
- Does **not** train an HMM yet (stubs in `train.py` and `eval.py`)

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
        }
    },
)
```

## Outputs

- `data/dataset_summary.json` (manifest/schema/roles/stats + split sizes)
