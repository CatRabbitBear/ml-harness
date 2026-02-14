# FX RV Regression

Regression plugin for FX realized-volatility experiments using `mlh-data` dataset artifacts.

Implemented experiments:
- `rv_base_persist_shift1`
- `rv_rms5_stats_ridge`, `rv_rms5_stats_gbr`
- `rv_rms5_vec8_ridge`, `rv_rms5_vec8_gbr`
- `rv_pca6_ridge`, `rv_pca6_gbr`
- `rv_pca6_abs12_ridge`, `rv_pca6_abs12_gbr`
- `rv_combo15_ridge`, `rv_combo15_gbr`
- `rv_stress1_ridge`, `rv_stress1_gbr`

Backward-compatible aliases:
- `rv_regress_v1_persist` -> `rv_base_persist_shift1`
- `rv_regress_v1_pca6abs_ridge` -> `rv_pca6_abs12_ridge`
- `rv_regress_v1_pca6abs_gbr` -> `rv_pca6_abs12_gbr`

## Dependencies

```bash
pip install -e ".[mlh-data,data,sklearn]"
```

## Notes

- Uses date split boundaries from plugin params:
  - train <= `split.train_end_date`
  - val in (`split.train_end_date`, `split.val_end_date`]
  - test in (`split.val_end_date`, `split.test_end_date`]
- Expects targets in `params.experiment.target_cols` to exist in dataset columns.
- PCA experiments use features `PC1..PC6` plus `PC1_abs..PC6_abs`.
- Logs top-quantile precision, recall, and lift.
- For trained experiments, also logs test deltas vs persistence.
- Writes per-horizon diagnostic plots under `plots/<target_col>/`.

## Example YAML (persist)

```yaml
plugin:
  key: fx.rv_regression

run:
  experiment: mlh-fx-rv
  seed: 42
  strict: true
  tags:
    purpose: realdata

data:
  dataset_id: fx:rv_dataset:v1
  dataset_path: ${DATASET_PATH}
  split_name: default

params:
  experiment:
    name: rv_base_persist_shift1
    target_cols:
      - rv_fwd5__mean
      - rv_fwd10__mean
      - rv_fwd20__mean
```
