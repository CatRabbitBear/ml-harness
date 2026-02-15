# FX Local Vol Regression

Regression plugin for FX local-vol experiments using `mlh-data` dataset artifacts.

Implemented experiments (condensed ladder):

- `local_base_zero`
- `local_base_shift1`
- `local_features_a_gbr` (local rms + global rms stats)
- `local_features_b_gbr` (+ local impulse features)
- `local_features_c_gbr` (+ broad latent + optional PCA features)
- `local_lat_hist_abs_gbr` (abs latent-return history block)
- `local_lat_hist_signed_gbr` (signed latent-return history block)
- `local_lat_hist_plus_global_gbr` (abs + signed history + global rms context)
- `local_lat_rms_nuanced_gbr` (abs latent + local/global rms shape features)

## Dependencies

```bash
pip install -e ".[mlh-data,data,sklearn]"
```

## Notes

- Uses date split boundaries from plugin params:
  - train <= `split.train_end_date`
  - val in (`split.train_end_date`, `split.val_end_date`]
  - test in (`split.val_end_date`, `split.test_end_date`]
- If `params.experiment.target_cols` is empty, targets are resolved from dataset `roles.json`.
- Logs RMSE/MAE/spearman, top-quantile metrics, bias metrics, and turning-point metrics.
- Writes per-target predictions, metrics, model/baseline artifacts, and diagnostic plots.
- Feature-set A/B/C columns are resolved from available dataset columns to support latent and PCA bundles.

Plugin key:

- `fx.local_vol_regression`
