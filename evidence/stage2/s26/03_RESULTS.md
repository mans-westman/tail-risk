# 03 — Results record (sanitized excerpt; full artifacts stay in the run directories)

Revision note: this record was revised after the trial. The run directories `outputs/` are
local and gitignored, so they are not a published link — the tables here are the published
record. The figures in `figures/` were regenerated from copies of the saved artifacts after a
reviewer-discovered defect in the curve figure; the trial's originals are preserved in
`figures/superseded/`. Scores and tables were never affected — the defect was confined to the
plotting selection. See `04_REPAIR.md`.

Both runs: synthetic GARCH(1,1)-Gaussian DGP, seeds 6201/6202, partitions train 3000 /
validation 1500, `historical_window` 1000, alpha grid `[0.005, 0.01, 0.025, 0.05, 0.1, 0.2]`,
no test split.

## 1. Execution

| | baseline (`gaussian.json`) | comparison (`gaussian_ewma.json`) |
|---|---|---|
| models | `historical_simulation`, `garch_student_t` | same + `example_ewma_gaussian` |
| forecast rows | 41,232 | 67,476 (adapter: 26,244) |
| offline checks | 32/32 passed | 32/32 passed |
| verify | **27/27 applicable passed; 0 failed; 0 unavailable; 0 n/a** | **33/33 applicable passed; 0 failed; 0 unavailable; 0 n/a** |
| verification coverage | `partial=False` | `partial=False` |
| failure rows / invalid outputs | 0 / 0 | 0 / 0 |
| figures | 3 PNGs | 3 PNGs |

The six additional checks in the comparison run are the custom-adapter ones:
`custom_factory_provenance`, `custom_factory_source:example_ewma_gaussian`,
`model_capabilities_match:example_ewma_gaussian`,
`forecast_keys_complete:example_ewma_gaussian`,
`checkpoint_files_match_manifest:example_ewma_gaussian`,
`checkpoint_replay:example_ewma_gaussian`.

Replay record: `compared=26244 mismatches=0 failed_rows=0; requested=26244 answered=26244`.
Manifest: `module_sha256=dc99ba8410c15ed1…`, checkpoint
`checkpoints/example_ewma_gaussian/checkpoint.json` (`383` bytes,
`sha256=8d8bd6860a525b8d…`).

**Baseline invariance.** The 24 metric rows for the two baselines are identical between the two
runs (`pandas.DataFrame.equals` → `True`). Adding the adapter changed neither the simulated
data nor the baseline forecasts or scores.

## 2. Matched sample counts and exclusions

`historical_simulation` declares `minimum_history=1001`; the other two declare 63. The common
mask is therefore set by the widest warm-up.

| split | n (all models) | available: ewma / garch | available: historical | masked out |
|---|---|---|---|---|
| train | 1,999 | 2,937 | 1,999 | 938 |
| validation | 499 | 1,437 | 499 | 938 |

Every model has `n_failed_rows=0`, `n_invalid_output=0`, and `n` is identical across models in
each cell. The 938 dropped origins per cell are the historical-simulation warm-up, not adapter
failures. `oracle_curves.csv` covers 44,964 oracle-matched rows; those are implementation
checks against simulated latent state, not forecasting skill.

## 3. Primary alpha (0.025) — mean FZ0 on the common mask

| split | `example_ewma_gaussian` | `garch_student_t` | `historical_simulation` |
|---|---|---|---|
| train | **-3.769819** (n=1999) | -3.783353 | -3.769025 |
| validation | **-3.751181** (n=499) | -3.754682 | -3.693883 |

Paired differences (`paired_scores.csv`, negative = better): train, ewma − garch = **+0.013534**
(ewma worse); validation, ewma − garch = **+0.003501** (ewma worse); ewma − historical =
−0.000794 (train) and −0.057298 (validation). Differences are descriptive: no interval, no
standard error, no significance marker.

## 4. Exceedances at 0.025

| split | model | observed | expected | Kupiec p | Christoffersen p |
|---|---|---|---|---|---|
| train | ewma | 48 | 49.98 | 0.776 | 0.881 |
| train | garch | 37 | 49.98 | 0.052 | 0.716 |
| train | historical | 51 | 49.98 | 0.884 | 0.049 |
| validation | ewma | 12 | 12.48 | 0.891 | 0.284 |
| validation | garch | 8 | 12.48 | 0.170 | 0.609 |
| validation | historical | 9 | 12.48 | 0.295 | 0.565 |

Limits that apply: Kupiec (unconditional coverage) and Christoffersen (first-order independence)
are the only backtests available; **there is no ES-specific backtest and no bootstrap
inference** — both unavailable, not passed. `undefined` tests are left undefined. Sparse counts
(e.g. 3 exceedances at 0.005 on validation) are shown as they are, and non-rejection at a sparse
alpha is not evidence of calibration. The validation split is the declared second partition of
the same simulated path, not an independent sample of market conditions.

These counts are on the **common mask**: 499 validation origins that every model could score.
The realized-return figure reports a different population — each model's **own available
rows** at 0.025 — so its panel for EWMA says 43 exceedances where this table says 12. Both are
correct for their sample: 43 on 1,437 available origins, 12 on the 499 matched ones. The same
gap appears for `garch_student_t` (34 available vs 8 matched) and for
`historical_simulation` (9 and 9, because its own warm-up already sets the mask). Neither number
is comparable to the other, and the figure states which one it draws.

## 5. Figures

`report --plots` wrote three PNGs during the trial. After review found a defect in the curve
figure, all three were regenerated from copies of the saved artifacts. The table describes the
files now in `figures/`; the trial's originals are kept in `figures/superseded/`.

| file | bytes | dimensions | sha256 (first 16) | relation to the trial original |
|---|---|---|---|---|
| `realized_var_es_primary_alpha.png` | 401,737 | 1540×1204 | `79dee8bea6cdafdb` | same plotted data; title and caption now name the population |
| `coverage_rate_vs_alpha.png` | 117,921 | 1008×910 | `2ad167b5eb462ff1` | byte-identical |
| `curves_last_common_origin.png` | 105,539 | 1008×644 | `aea2d782753bdf6b` | one series selected instead of two |

Superseded originals (`figures/superseded/`), kept so the defect stays inspectable:

| file | bytes | sha256 (first 16) | defect |
|---|---|---|---|
| `curves_last_common_origin.png` | 107,810 | `4cddfa846c34b897` | 36 rows at origin 1498, 18 duplicate `(model, alpha)` pairs |
| `realized_var_es_primary_alpha.png` | 399,672 | `f18921bcb1b0b759` | plotted data correct; caption silent on population |
| `coverage_rate_vs_alpha.png` | 117,921 | `2ad167b5eb462ff1` | none |

What was wrong, in numbers. Origin labels belong to a path: `example_garch_gaussian_train` and
`example_garch_gaussian_validation` are both indexed up to 1498. Selecting `origin_date ==
1498` returned 36 rows for three models, two per alpha per model, so each "curve" zig-zagged
between a train forecast and a validation forecast under a caption that named only validation.
The corrected selection returns 18 rows, exactly one per `(model, alpha)`, all from
`example_garch_gaussian_validation`. The baseline run had the same defect at 24 rows / 12
duplicates and now has 12 rows.

Regeneration provenance: each run directory was copied to `.tmp/s26_regen/{baseline,ewma}`
and `PYTHONPATH=src .venv/bin/python -m tailrisk report --run <copy> --plots` was run there.
`report` reads saved CSVs; it does not fit, score or rewrite them. Verified by hashing the five
numerical artifacts before and after — `forecasts.csv`, `metrics.csv`,
`score_decomposition.csv`, `paired_scores.csv` and `curve_diagnostics.csv` are byte-identical
between the original run directory and the regenerated copy.

Visual inspection: **not performed.** This runtime disables image bytes on the file-read path
and exposes no OCR or visual-question tool, so no claim of sight is made here. What was checked
instead: each PNG opens and passes PIL `verify()` with the dimensions above; and the renderer
was instrumented on the real artifacts, recording that it hands matplotlib six calls of six
points each — one VaR and one ES line per configured model, every x value in the configured
alpha grid. That is a check on the data the figure draws, not on how it looks.


## 6. What this record does and does not establish

- Establishes: the supplied EWMA–Gaussian function can be integrated through the public adapter
  contract, runs to completion beside the baselines, passes every applicable verifier check,
  and replays from a checkpoint with no fitting step.
- Establishes: the numbers the adapter reports are the numbers the supplied function computes
  (adapter test A) and are reproducible from the saved settings (verify replay, 26,244/26,244).
- Does not establish: superiority, significance, regulatory suitability, or leakage-freedom. One
  synthetic series, one seed, whole-method differences without intervals; the factory is
  trusted local code and no universal leakage guarantee exists.
- Does not establish anything about markets: the data are simulated.

Label: **fresh-agent self-reproduction**, not independent external validation.
