# Local Vol Plugin Bootstrap

This document captures what we learned from the global-vol (`fx_rv_regression`) work and how to use it to start a separate local-vol plugin without dragging the global assumptions into the new task.

## Why a New Plugin

A new plugin is the right move for local-vol experiments because:

- label semantics are different (`local` and `excess_local` targets vs global ignition),
- feature-set growth is larger (A/B/C sets across latent and PCA variants),
- the run matrix can explode quickly, and
- we want global and local results separated in MLflow and code history.

Recommended new plugin key pattern:

- `fx.local_vol_regression` (example)

Keep existing global plugin (`fx.rv_regression`) stable as historical baseline.

## What We Learned in Global Vol

From the global ignite runs:

- `shift+1` baseline is hard to beat on RMSE because it tracks continuity.
- `rms5` global stats had better turning behavior than naive baselines.
- PCA and latent sets were additive at best in the global target setting.
- tail amplitude remained hard; many models under-shot highs/lows.

Implication for local plugin:

- baseline choice and turning metrics matter as much as RMSE.
- evaluate "level fit" and "turn timing" separately.
- start with a condensed ladder first, then expand.

## Baseline Semantics to Preserve

In global work we had two useful baselines:

- `base_zero`: `y_hat = 0` (no-change in ignite space)
- `base_shift1`: `y_hat_t = y_{t-1}`

For local plugin, keep both. Treat:

- `shift+1` as persistence benchmark,
- `zero` as no-change benchmark in normalized label space.

## Condensed Ladder (Recommended v1)

To avoid combinatoric blowup, start with a condensed set:

1. `base_shift1`
2. `features_A_gbr`
3. `features_B_gbr`
4. `features_C_gbr`

Run this for each representation family (latent, PCA) separately.

### Suggested Feature Set Mapping

Define one explicit A/B/C contract and keep it identical between latent and PCA families where possible.

Example:

- `A`: local state minimal (few strongest features)
- `B`: + local impulse / cross-sectional structure
- `C`: full candidate set for that family

Use one model first:

- `GradientBoostingRegressor` with fixed hyperparameters

Add extra models only after A/B/C findings are clear.

## Metrics to Keep (No Changes)

Carry over these metrics from global plugin:

- `mae`, `rmse`
- `topq_precision`, `topq_recall`, `topq_lift`
- bias metrics:
  - `mean_pred`, `mean_true`, `mean_error`, `mean_error_topq`
- turning metrics:
  - `turn_precision`, `turn_recall`, `turn_f1`
  - `turn_rate_true`, `turn_rate_pred`
  - `delta_sign_acc`

These already proved useful for separating "tracks level" from "captures turns".

## Local-Vol Labeling Guidance

For local plugin, define labels explicitly in dataset manifest/spec:

- `igniteH__CCY = log(rv_fwdH__CCY + eps) - log(rms5__CCY + eps)`
- optional `excess_igniteH__CCY = igniteH__CCY - igniteH__global`

Do not mix local and global target families in one plugin at first.

## Reuse Map From Current Plugin

The following files are good direct reuse candidates (copy then rename):

- `plugins/fx_rv_regression/data.py`
  - robust date parsing and deterministic split logic.
- `plugins/fx_rv_regression/eval.py`
  - complete metric block including turning metrics.
- `plugins/fx_rv_regression/plots.py`
  - high-signal diagnostics (overlay, events, scatter, residual).
- `plugins/fx_rv_regression/artifacts.py`
  - summary/metrics/predictions/model artifact writers.
- `plugins/fx_rv_regression/plugin.py`
  - thin orchestration shape and tracking conventions.

Parts to rewrite for local plugin:

- `plugins/fx_rv_regression/config.py`
  - experiment names, target defaults, feature family config.
- `plugins/fx_rv_regression/train.py`
  - feature plans A/B/C and family-specific columns.
- `apps/fx_rv_regression_run.py`
  - new plugin key, dataset id, env names.
- `apps/fx_rv_regression_suite_run.py`
  - condensed local ladder list.

## Practical Implementation Plan (New Local Plugin)

1. Create `plugins/fx_local_vol_regression/` from `fx_rv_regression` skeleton.
2. Replace experiment plan logic with local A/B/C definitions for:
   - latent family,
   - PCA family.
3. Keep evaluator and plots unchanged initially.
4. Add two suite runners:
   - latent suite (shift+1 + A/B/C)
   - PCA suite (shift+1 + A/B/C)
5. Compare in MLflow by:
   - avg RMSE,
   - avg turning F1,
   - avg topq precision.

## Guardrails

- Freeze split boundaries across all local experiments.
- Keep dataset manifests explicit about label formula and eps.
- Do not change model hyperparameters while comparing latent vs PCA family.
- Only expand to larger model sets after condensed ladder conclusions are stable.

## Definition of "Done" for Local v1

Local plugin v1 is good enough when:

- both latent and PCA condensed ladders run end-to-end,
- shift+1 benchmark is included in each family,
- turning metrics are logged and compared,
- one clear recommendation emerges for which family to expand first.
