# Methods, attribution and limitations

This page states what the toolkit computes, in enough detail to check the code against the
formulas, and what the results may not be used to claim. It matches `src/tailrisk/`; where
the page and the code disagree, the code is right and this page is a bug. Installation and
commands are in the top-level `README.md`, the data input in
[DATA_CONTRACT.md](DATA_CONTRACT.md), the model interface in
[MODEL_ADAPTER.md](MODEL_ADAPTER.md), and the report in [REPORTING.md](REPORTING.md).

## Scope

One return series, one session ahead. At origin `t` a model sees returns up to and including
`t` and emits a value-at-risk (`var`) and an expected shortfall (`es`) for the return over
`t → t+1`. `alpha` is a lower-tail probability, and returns are signed decimal log returns,
so a loss is negative and VaR/ES are negative numbers. The supported domain is
`es <= var < 0`; a successful forecast outside it is counted as an invalid output rather
than clipped or repaired.

This is a comparison and evidence toolkit. It does not forecast live, size positions, or
certify a model.

## Timing and splits

- A split is a range of **target** dates. Origins may lie in the preceding period, so the
  first scored origin of a split is not cropped for lack of history.
- Splits are executed in the order train, validation, test, and only those named in the
  configuration are executed. A configuration that names no test split never simulates,
  fits on, or scores a test observation.
- Every scored origin needs a full trailing window: `max(window + 1, 63)` returns for
  historical simulation and 63 for GARCH-t and the MLP. Origins short of that are excluded
  by declaration and counted, not silently dropped.
- Each model's prediction is a function of the causal prefix handed to it. The runner
  constructs that prefix as an owning copy ending at the origin; no adapter receives a
  target value, an oracle, a latent volatility path, or a view that exposes later rows.
- A fitted GARCH-t is fitted once, on the train split, and frozen for every split the run
  executes. The MLP fits on train and selects its checkpoint on validation. Train-period
  scores are in-sample diagnostics; validation-selected forecasts are development evidence,
  not a sequence of genuinely out-of-sample forecasts from the fitted period.

## Estimators

**Historical simulation** (`historical_simulation`). Order the trailing `n` returns ascending
and let `b = n * alpha`. With `b > 1`, `var` is the observation at index `ceil(b) - 1` and
`es` is the tail mean with the boundary observation weighted by its fractional mass:

```
full    = floor(b),  f = b - full
q       = ordered[ceil(b) - 1]
tail    = sum(ordered[:full]) + f * q
es      = tail / b
```

With `b <= 1` the tail contains at most one observation, so the tail mean *is* that
observation and both `var` and `es` are the window minimum. Evaluating the general formula
there would return `es` one ulp above `var` for a row that is numerically correct, which the
contract reports as an invalid output; the early return removes that artefact and leaves the
formula above unchanged where it averages more than one observation. No fitted parameters
and no distributional assumption.

**GARCH(1,1)-Student-t** (`garch_student_t`). `arch`'s `ZeroMean` with `GARCH(1,0,1)` and a
`StudentsT` innovation, fitted by maximum likelihood on train returns rescaled by 100 for
numerical stability, then converted back to decimal units (`omega / 100**2`; `alpha[1]`,
`beta[1]` and `nu` unchanged). Prediction filters the variance recursion from the
unconditional variance using the supplied prefix, then maps the filtered scale through the
Student-t quantile and tail-mean formulas. Fitting requires at least 64 returns;
`verify` refits independently from the panel's train slice and replays every saved forecast
instead of trusting the saved parameters.

**Joint MLP** (`joint_mlp`, optional). Six causal features at each origin — scaled return,
its lag, 5/21 and 21/63 relative RMS ratios, downside RMS, and 63-return excess kurtosis —
fed to a small network that emits `(var, es)` jointly for the requested alpha levels,
trained directly on the FZ0 loss. Feature standardization uses train statistics only.
Checkpoint selection is on validation loss; the selected checkpoint is written as JSON plus
NPZ and replayed by `verify` without refitting. Origins whose feature window is incomplete
are recorded as explicit failed rows. Installing the `neural` extra is required only when
`joint_mlp` is configured.

## Scoring

Forecasts are scored with the FZ0 joint VaR/ES loss (Fissler–Ziegel; Patton–Ziegel–Chen,
eq. 6). For `e < 0`:

```
L(y, v, e, alpha) = -1{y <= v} (v - y) / (alpha e) + v / e + log(-e) - 1
```

Lower is better. Because the indicator means `1{y <= v} = 1{v - y >= 0}`, the loss splits
exactly into

```
shortfall = -max(v - y, 0) / (alpha e)      ratio = v / e - 1      log = log(-e)
```

and `mean_fz0 = mean_shortfall + mean_ratio + mean_log` on the same rows. The residual of
that identity is written to `score_decomposition.csv` and a value beyond float64 tolerance
fails `evaluate`, because a broken identity is a coding defect.

Two invariants keep the comparison honest:

- **One mask.** Within each `(series, split, alpha)` cell every model is scored on the same
  origins — the intersection of admissible rows — so `n` is identical across models and
  means are comparable. A model with no admissible row in a cell empties the cell
  (`n = 0`) rather than letting the survivors be scored alone.
- **No repair.** A non-finite or out-of-domain forecast is counted in `n_invalid_output`
  and explained in `failures.csv`; it is never clipped, sign-flipped, sorted or imputed.
  Coverage cost stays visible in `n_available`, `n_failed_rows` and `n_dropped_by_mask`.

`paired_scores.csv` reports mean FZ0 differences between a model and each conventional
baseline present, on that same mask. No standard error, confidence interval or significance
star is produced: `n` is a count of dependent, overlapping origins, not independent samples.

## Exceedance diagnostics

Exceedances are `1{y <= var}`. The rate is reported with observed and expected counts.
Kupiec's unconditional-coverage test and Christoffersen's independence test are computed as
likelihood-ratio statistics against their chi-square references. A statistic is `nan` when it
is undefined — for example no exceedances at all leaves the unconditional test without a
finite alternative — and `nan` is reported as undefined, never imputed. `n_transitions`
records how many adjacent pairs entered the independence test and `n_chain_breaks` how many
were severed by a scoring-mask gap, so non-adjacent days are never silently joined.

There is no ES-specific backtest and no bootstrap inference in this version; both are
unavailable, not passed.

## Synthetic generators and oracles

`dgp` configs simulate a GARCH(1,1) recursion with Gaussian or unit-variance Student-t
innovations, discarding a burn-in first. The generator returns the latent `sigma` path and
the analytical `var`/`es` implied by the innovation distribution, which is what
`oracle_curves.csv` holds. The oracle is evaluation-only: it is never a model input, and a
real-data panel has no oracle at all, so `verify` reports oracle comparisons as *not
applicable* rather than passed.

Oracle comparisons check that an implementation recovers a known conditional risk; they are
not evidence of forecasting skill.

## Verification

`verify` re-derives artifacts from the inputs rather than reading them back and agreeing
with itself: forecast-key completeness from the permitted origins and declared minimum
histories, target ordering, unit ordering, fitted-parameter provenance, exact failure
reasons, and — where a model supports it — deterministic replay (GARCH-t refit, MLP
checkpoint replay, custom-adapter `checkpoint` mode). `verification_coverage.json` lists
every check with `passed`, `failed`, `unavailable` or `not_applicable` and a reason.
Unavailable optional replay is reported as partial coverage, never as a pass. No check
establishes "no leakage in general".

## Attribution

Methods:

- Fissler, T. and Ziegel, J. F. (2016). Higher order elicitability and Osband's principle.
  *Annals of Statistics* 44(4), 1680–1707 — the joint VaR/ES scoring function.
- Patton, A. J., Ziegel, J. F. and Chen, R. (2019). Dynamic semiparametric models for
  expected shortfall (and Value-at-Risk). *Journal of Econometrics* 211(2), 388–413 —
  FZ0 in the form used here.
- Bollerslev, T. (1986, 1987) — GARCH and the Student-t GARCH innovation.
- Kupiec, P. (1995) — unconditional coverage; Christoffersen, P. F. (1998) — interval
  forecast independence.

Software, at the versions recorded in each run's `manifest.json`:

| Package | Used for | License declared by the installed distribution |
|---|---|---|
| NumPy | arrays, quantiles | BSD-3-Clause (plus bundled 0BSD/MIT/Zlib/CC0-1.0 components) |
| SciPy | distributions (`norm`, `t`), optimization | BSD-3-Clause |
| pandas | panel, artifact tables | BSD-3-Clause |
| `arch` | GARCH-t maximum likelihood | NCSA |
| Matplotlib | optional figures | PSF-style Matplotlib licence |
| JAX, Optax | optional MLP training | Apache-2.0 |
| pytest | tests | MIT |

Those are the licences the installed packages declare; they are not a legal determination
for redistribution, and the list is not a substitute for reading each licence.

**No project licence has been selected.** Choosing one is a prerequisite for publication and
is deliberately left to the project owner. Until then this repository grants no rights beyond
what its dependencies' licences and applicable law allow, and market data is not
redistributed (`data/` is ignored and `ingest` re-acquires from a config).

## What is not claimed

- No universal leakage certificate. The checks above are the ones that ran; restricted
  arguments cannot prove that trusted local plugin code never reads a file, and no check
  detects contaminated pretraining or researcher selection.
- No model superiority. Comparisons are whole-method; no matched-architecture or controlled
  claim is made, and a lower mean FZ0 on one dataset is not a ranking.
- No statistical significance. Differences carry no interval, and repeated seeds, overlapping
  windows and one series are not independent evidence.
- No ES calibration claim, no regulatory-suitability or capital-adequacy claim, and no
  interpretation of any loss term as a cost of capital.
- Synthetic runs are execution and implementation checks with limited tail counts, not
  performance evidence, and oracle accuracy is not forecasting skill.
- Curve crossings are a finite-grid diagnostic on complete per-origin grids; they are not a
  proof of monotonicity or of the integrated-quantile identity.
