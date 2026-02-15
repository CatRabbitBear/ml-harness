# FX RV Regression

Regression plugin for FX ignition experiments using `mlh-data` dataset artifacts.

Implemented experiments (trimmed ladder):
- `ignite_base_zero`
- `ignite_base_shift1`
- `ignite_rms5_stats_gbr`
- `ignite_pca6_gbr`
- `ignite_pca6_abs12_gbr`
- `ignite_combo15_gbr`

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
- Default targets are ignite labels: `ignite5`, `ignite10`, `ignite20`.
- Ignite targets are already log-style labels, so default is `preprocess.log_target: false`.
- Logs top-quantile precision/recall/lift plus bias metrics (`mean_pred`, `mean_true`, `mean_error`, `mean_error_topq`).
- For non-zero experiments, logs test deltas vs zero baseline.
- Writes per-horizon diagnostic plots under `plots/<target_col>/`.

## Example YAML

```yaml
plugin:
  key: fx.rv_regression

run:
  experiment: mlh-fx-ignite
  seed: 42
  strict: true
  tags:
    purpose: realdata

data:
  dataset_id: fx:ignite_dataset:v1
  dataset_path: ${DATASET_PATH}
  split_name: default

params:
  experiment:
    name: ignite_combo15_gbr
    target_cols:
      - ignite5
      - ignite10
      - ignite20
```
