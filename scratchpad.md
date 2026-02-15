# A. New plugin + dataset naming scheme

### New plugin key

- `ignite_local_ccy_v1` (or similar)
  Keep your current one as `ignite_global_v1`.

### Datasets (two bundles)

1. `fx_geom_localvol_latent_v1`
   **No PCA features** (pure latent + global stats)
2. `fx_geom_localvol_pca_v1`
   Includes PCA features (frozen on train)

Both datasets must share:

- same date range
- same splits
- same label definitions
- same base columns (timestamp, lat_ret\_\_)

That makes comparison airtight.

---

# B. Labels: local + “deglobbed” local

This is the key to separating local vs global. You don’t remove global; you **model it explicitly**.

Pick one currency first (JPY), then USD later.

## B1) Local vol level (easy sanity)

For currency C and horizon H:

- `rv_fwdH__C`

This verifies your plumbing and whether “local persistence” is strong.

## B2) Local ignition (what you actually care about)

- `igniteH__C = log(rv_fwdH__C + eps) - log(rms5__C + eps)`

Same idea as global ignite, but per currency.

## B3) Local-only (remove global component without brittle filtering)

This is the big one.

Define:

- `rv_fwdH__global = mean over currencies (rv_fwdH__*)`
- `rms5__global = mean over currencies (rms5__*)`

Then create “excess local ignition”:

### Option 1 (difference of ignitions)

- `excess_igniteH__C = igniteH__C - igniteH__global`

Interpretation:

- “Is C igniting more than the market?”

This is **not brittle**, and it directly neutralises global heat.

### Option 2 (ratio of levels)

- `excess_log_rvH__C = log(rv_fwdH__C+eps) - log(rv_fwdH__global+eps)`

Interpretation:

- “Is C’s forward vol higher than global forward vol?”

Both are good. I’d start with **Option 1** because it matches your current framework.

> This is your clean separation mechanism. No hard filters, no survivorship, no ‘only calm days’. Just a relative signal.

---

# C. Features: minimal ladders that won’t drown you in noise

You’re right to worry “PC1..PC6 might just be noise”. So do ablations that _force_ PCA to prove itself.

## C1) Baseline (local state + global context)

For currency C:

- `rms5__C`
- `rms5__global_mean` (your current `rms5__mean`)
- `rms5__global_std`
- `rms5__global_max`

That answers: “Does current state already tell us everything?”

## C2) Latent impulse (local + global)

- `abs(lat_ret__C)`
- `abs_latent_mean` (mean over currencies)
- `max_abs_latent`

This catches “something just happened”.

## C3) PCA features, but don’t start with 6 PCs blindly

Do **three PCA feature sets**, in increasing capacity:

### PCA set A (single-axis best-of)

- `abs(PC1)` only (or `PC1` + `abs(PC1)`)

### PCA set B (top-2)

- `PC1, PC2, abs(PC1), abs(PC2)`

### PCA set C (full)

- `PC1..PC6 + abs(...)`

This prevents you discarding PCA early. If PCA is useful, you’ll often see it with 1–2 components first.

**Important:** make sure PCA is _fit on train only_, frozen.

---

# D. Two PCA pipelines to compare (don’t conflate)

PCA can mean two different things here. You should test both explicitly.

## D1) Vol-shape PCA (what you’ve done)

Input vector per day:

- `rel5__C = rms5__C - mean(rms5__)`

Good for “who is hot relative to others”.

## D2) Return-shape PCA (new, likely better for local shocks)

Input vector per day:

- `shock__C = abs(lat_ret__C) - mean(abs(lat_ret__))`

Good for “who moved today”, cross-sectional dispersion.

I agree with your gut: **return-shape PCA** might be where the juice is for local shocks.

---

# E. Models: keep it boring (for now)

To avoid overfitting and “hyperparameter archaeology”:

- **GBR** shallow (same settings you already use)
- Optional: **ExtraTreesRegressor** as a robustness check (often good for interactions)
- Keep seeds fixed.

---

# F. Metrics & diagnostics (so you don’t fool yourself)

For each label type (local, ignite, excess_ignite):

## Core metrics

- RMSE / MAE (on label space)
- Spearman
- Top-20% precision/recall/lift (on **positive tail**)

## The separation test (must-have)

For currency C:

- correlation of prediction with `ignite_global`
- correlation of prediction with `excess_igniteC`

You want:

- high corr with `excess_igniteC`
- _not just_ high corr with global

Also: evaluate performance on **days where global ignite is small**:

- Filter to `|ignite_global| < median` or below 60th percentile
- Then measure top-quantile recall on `excess_igniteC`

That directly checks “local signal when global is quiet”.

---

# G. Integrity safeguards (so norms don’t creep)

Here’s the checklist that prevents silent drift:

1. **Dataset manifest must record**
   - eps
   - window definitions
   - “inclusive/exclusive indexing”
   - currency order
   - split boundaries
   - PCA fit range + scaler params hash

2. **One source of truth for feature math**
   - put RMS, forward RV, ignite label functions in a shared module used by both dataset builders

3. **Golden-row unit test**
   - pick one timestamp
   - assert exact values for:
     - rms5\_\_C
     - rv_fwd10\_\_C
     - ignite10\_\_C
     - excess_ignite10\_\_C
       This catches off-by-one and wrong window bugs immediately.

4. **Transform artifacts versioned**
   - `pca_transform.joblib` stored alongside dataset
   - include SHA256 of transform in manifest

---

# H. The smallest “v1” that answers the question

Start with **JPY only**, horizons {10,20}, and label `excess_igniteH__JPY`.

Run these comparisons:

1. Baseline: local+global RMS stats
2. - latent impulse
3. - vol-shape PCA set A/B/C
4. - return-shape PCA set A/B/C

If PCA helps, it will show here.

---

## More discussion at a high level

Yep, Ant — that’s a fair breakdown. A couple of small tweaks to make it watertight:

* We showed **global ignition** is mostly captured by **global state features** (your RMS stats), and PCA/latent are at best additive there.
* Now we’re asking a different question: **can features separate “local ignition” from “global heat”?** That’s where PCA might finally earn its keep.

Plain-English restatement, tightened:

1. **What we’ve done (global):**
   We built models that can predict **global ignition** (vol expansion/contraction) using features derived from latent returns and/or PCA. The best directional/turning behaviour came largely from **global RMS statistics**, with PCA not clearly dominating.

2. **What we’re doing next (local):**
   We’re switching the label from “global ignition” to **currency-specific ignition** (e.g., JPY ignition). Then we’ll test whether **latent vs PCA** features can predict that.

3. **How we stop the model cheating by just learning global vol:**
   In at least one key experiment, we’ll use a **“local minus global”** target (your excess idea), so the model is rewarded for predicting **JPY ignition that is unusual relative to global conditions**, not just “everything is hot”.

4. **About revisiting PCA without RMS/demeaning:**
   Totally reasonable. You’re separating two concepts that were getting tangled:

   * PCA built on **vol-shape** (RMS-based, cross-sectional demeaned)
   * PCA built on **return-shape** (instantaneous dispersion, possibly using abs returns, optionally demeaned)

   The goal is to make the pipeline easier to reason about and reduce “double normalization” confusion.

One nuance: even in the new PCA variants, **some** centering/standardising is still normal (PCA basically expects it), but you’re right to avoid stacking multiple “demeanings” that make the meaning of a value opaque.

So yes: your breakdown is solid — we’re moving from “global ignition is predictable” to “can we isolate local ignition beyond global conditions, and does PCA help more than raw latent features when we force it to?”
