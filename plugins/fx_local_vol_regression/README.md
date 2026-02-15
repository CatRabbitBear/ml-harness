# FX Local Vol Regression (Scaffold)

This plugin is a scaffold for upcoming local-vol experiments (currency-local labels and A/B/C feature sets).

Current status:

- Loads and validates dataset artifacts via `mlh_data`.
- Resolves target columns from `params.experiment.target_cols` or dataset `roles.json`.
- Logs a bootstrap summary artifact (`reports/bootstrap_summary.json`).
- Does **not** train models yet.

Intended next step:

- Implement condensed ladder:
  - `local_base_shift1`
  - `local_features_a_gbr`
  - `local_features_b_gbr`
  - `local_features_c_gbr`

Plugin key:

- `fx.local_vol_regression`
