# Experiment protocol — 2026-11-04 (frozen before execution)

Bounded development experiment closing baseline Stage 1: compare the alpha-conditioned joint
MLP with historical simulation and GARCH–Student-t on identical eligible **validation** rows
in original return units. This document is written and dated before any run of this
experiment. It reuses the implemented pipeline unchanged: no new model, no architecture,
feature, loss, trainer-default or checkpoint-selection change.

## 1. Scope

- Three cases: one Gaussian GARCH(1,1) simulation, one standardized Student-t GARCH(1,1)
  simulation, and the existing single-series real S&P 500 train/validation period.
- Three fixed model (initialization/shuffle) seeds per case; **all** are reported.
- Validation only. No test-period simulation, download, forecast or score. The synthetic
  test partitions are not even declared, so no test array exists.
- No hyperparameter search, no learning-rate fallback, no score-driven rerun, no seed
  selection. Underperformance is a reportable result.

## 2. Window decision: W = 1000 (guide §6.1 default), not W = 100

Decided from sample availability and tail resolution only. No performance score of any
model, at any window, was inspected before this decision.

**Tail resolution of the historical-simulation estimator.** With the guide's
generalized-inverse quantile and fractional-boundary ES, `m = W · alpha` is the number of
window observations the tail actually integrates:

| alpha | W=100: m | W=1000: m |
|---|---|---|
| 0.005 | 0.5 | 5 |
| 0.010 | 1.0 | 10 |
| 0.025 | 2.5 | 25 |
| 0.050 | 5 | 50 |
| 0.100 | 10 | 100 |
| 0.200 | 20 | 200 |

At W=100 the two smallest levels have `m <= 1`: the quantile is the single minimum
observation and the ES integrates that one observation, so `VaR = ES = min(window)` by
construction. Two of six levels would be definitionally degenerate for the comparator, and
`es <= var` would hold as equality rather than as a fitted property. W=1000 removes the
degeneracy at every configured level (`m >= 5`).

**Sample availability at W=1000.** Each historical-simulation forecast needs `W` trailing
returns inside its own split array (a synthetic partition is an independent path; a panel
split carries permitted context):

- Real panel, training: 6,552 training targets; with `required_context = 1000` the first
  scored target is the 1,001st session, leaving **5,551** scored training origins (W=100
  would give 6,451). Cost: 900 training origins, ~14%, no validation loss.
- Real panel, validation: 1,006 targets; the panel attaches 1,101 preceding returns, so all
  **1,006** validation origins keep a complete 1,000-return window. No coverage loss.
- Synthetic partitions: train 10,000 returns → 9,937 MLP-eligible origins; validation 4,000
  returns → **2,999** common-mask origins after the 1,000-return window (7.5 -> 15 expected
  exceedances at alpha=0.005 versus the guide's 2,500-return default). The validation path
  is lengthened to 4,000 *before* execution to offset the window, which is a sample-size
  choice, not a score-driven one.

Decision: **W = 1000** everywhere (comparator lookback and the real panel's
`required_context`), because it keeps every configured alpha level non-degenerate on a
sample that is available. The real panel is re-ingested from the same dated raw snapshot
under a new directory; the existing `data/sp500/` snapshot and every existing run stay
untouched. A run at W=100 must not be compared with these runs without restating the
window, and the guide-vs-development window deviation recorded in STATUS.md is closed by
this choice for all subsequent work.

**The four lengths are distinct and couple only through the mask.** (1) *Historical
lookback* `W = 1000`: the comparator's rolling window size. (2) *Feature warm-up* 63: the
MLP's six features need 63 returns and GARCH starts at origin 62. (3) *Training
eligibility*: MLP fitting uses every scored train origin with a full feature window (real:
5,551; synthetic: 9,937), which at W=1000 is also the historical-comparator start because
the panel marks a train origin scored only once `required_context` returns precede it. (4)
*Common evaluation mask*: the evaluator intersects models per `(series, alpha)` cell, so the
binding start is `max(1000, 62, 62) = 1000` and W alone sets how many validation origins are
scored.

**Correction (2026-11-04, after the runs; no rerun):** the coupling was described here as
raising `required_context` to `W` removing the first 900 real training origins from *every
model's fitting array*. That is wrong for the conventional models. The panel's `is_scored`
flag is one boolean per origin, and it governs two things: which origins a model may
**forecast** at (`allowed_origins`), and which origins are **training-eligible for the joint
MLP**. It does not touch the GARCH–t likelihood sample: `_garch_rows` fits
`model.fit(train.returns)` on the whole train split array — 6,553 returns for this panel
(6,552 in-split targets plus the attached origin return), 1,000 of which are origins marked
unscored — so the GARCH fit sample is the same at W=100 and at W=1000. Historical simulation
is unaffected as well: its window simply needs 1,000 trailing returns. The real cost of
W=1000 is therefore 900 *forecast* origins per split and 900 *MLP training* origins, not
900 GARCH-fitting observations. A per-model eligibility field would still be needed to let
the MLP train on origins no model forecasts, but no model's fit is currently truncated by
`required_context`.

## 3. Datasets and frozen cutoffs

| Case | id | Source | Train | Validation | Data seeds |
|---|---|---|---|---|---|
| SYN-G | `exp_gaussian_20261104` | zero-mean GARCH(1,1), Gaussian innovations | 10,000 | 4,000 | 5101 / 5102 |
| SYN-T | `exp_student_t_20261104` | zero-mean GARCH(1,1), standardized Student-t, nu = 5 | 10,000 | 4,000 | 5201 / 5202 |
| REAL | `sp500_price_index` | `^GSPC` price-index close, decimal log returns, snapshot `data/sp500/raw_yahoo_chart.json` | 6,552 targets to 2015-12-31 | 1,006 targets 2016-01-04 … 2019-12-31 | seeded by the panel |

GARCH parameters (independent starter, guide §4.2, not paper settings): `omega = 5e-6`,
`a = 0.05`, `b = 0.90`, `initial_variance = 1e-4`, `burn_in = 2000`, `nu = 5`. Each split is
an independent path with its own seed and burn-in; no state is shared across splits.

Real-data cutoffs are frozen and unchanged from the ingest snapshot: `train_end =
2015-12-31`, `validation_end = 2019-12-31`. Cutoffs are **not** moved.

**The test period (2020–2025) is not used in this experiment**: it is not simulated,
forecast or scored here, and the synthetic test partitions are not even declared. It is
**not** an untouched holdout, and this document does not claim that. `outputs/sp500-real-20261102`
scored the full period with historical simulation and GARCH–t and its Kupiec/Christoffersen
diagnostics were computed and inspected; that exposure is recorded in STATUS.md and
`outputs/README.md` and is unaffected by this experiment. No joint-MLP forecast for any test
session has ever been produced. Note that the `exposure` string frozen into the ingest
configs and every run manifest is the declaration made at ingest time and understates this
history; it must not be quoted as the exposure record.

## 4. Models and fitting policy (unchanged)

- `historical_simulation`: trailing window W = 1000, guide §6.1 quantile/ES estimator.
- `garch_student_t`: zero-mean GARCH(1,1) with standardized Student-t, fitted once on the
  training split, parameters frozen for validation; causal variance filtering inside the
  validation array.
- `joint_mlp`: six engineered features + `log(alpha)`, 32/32 tanh, local-ordering
  parameterization, FZ0 objective, train-only feature standardization.
  Trainer defaults, unmodified: Adam `lr = 1e-3`, batch 256 origins, gradient-norm clip
  5.0, `max_epochs = 100`, validation patience 10, checkpoint selected by **original-unit**
  validation mean FZ0. Epoch 0 (initialization) is a legitimate checkpoint.

Alpha grid (unchanged): `[0.005, 0.01, 0.025, 0.05, 0.1, 0.2]`.

## 5. Seeds (listed before execution)

Model seeds, identical set for every case: **101, 102, 103** (initialization and batch
shuffle; the only randomness in the trainer). Frozen into each resolved configuration, so
the config hash covers them. Empty model seed for the two conventional models, which have
none. No seed is added, replaced or dropped after a score is seen.

## 6. Run plan

Nine runs, one per (case, model seed), each a new directory
`outputs/exp-20261104-<case>-s<seed>/`:

| Config | Output |
|---|---|
| `configs/experiment-20261104/syng_seed101.json` | `outputs/exp-20261104-syng-s101` |
| `configs/experiment-20261104/syng_seed102.json` | `outputs/exp-20261104-syng-s102` |
| `configs/experiment-20261104/syng_seed103.json` | `outputs/exp-20261104-syng-s103` |
| `configs/experiment-20261104/synt_seed101.json` | `outputs/exp-20261104-synt-s101` |
| `configs/experiment-20261104/synt_seed102.json` | `outputs/exp-20261104-synt-s102` |
| `configs/experiment-20261104/synt_seed103.json` | `outputs/exp-20261104-synt-s103` |
| `configs/experiment-20261104/real_seed101.json` | `outputs/exp-20261104-real-s101` |
| `configs/experiment-20261104/real_seed102.json` | `outputs/exp-20261104-real-s102` |
| `configs/experiment-20261104/real_seed103.json` | `outputs/exp-20261104-real-s103` |

Each run executes `run` → `verify` → `evaluate` → `report`. Scores are interpreted only
after `verify` passes, including the checkpoint replay of every MLP row.

## 7. Comparison rules

- **Rows:** validation split only, `status == "ok"`, on the evaluator's common mask per
  `(series, alpha)` cell, so `n` is identical across the three models within a cell.
- **Units:** original decimal log returns; no rescaled or percentage presentation except
  where explicitly labelled.
- **Primary development metric:** mean FZ0 at alpha = 0.025 (guide §7.2), per seed and
  pooled as the mean over the three seeds with the per-seed spread shown.
- **Secondary:** mean FZ0 across the full grid, equal-weight mean over the six levels;
  mean quantile loss; exceedance count/rate versus `n · alpha`; Kupiec and Christoffersen
  (with transition counts and chain breaks); oracle VaR/ES MAE and RMSE for the synthetic
  cases (oracle fields absent by construction for the real case). ES diagnostics are the
  existing exceedance-conditioned residual columns plus the FZ0 decomposition; no new
  statistic is invented here.
- **Coverage and failures** are reported for every cell: `n_available`, `n_failed_rows`,
  `n_invalid_output`, `n_dropped_by_mask`.
- **Seed variability:** the three model seeds share the same data, so seed spread measures
  initialization/shuffle variability only, not market or path variability. Seeds are not
  independent observations of the market; means over seeds are reported as a summary, never
  as significance.
- **No selection:** the best seed is not chosen, no model is selected for the next stage on
  these numbers, and no rerun is triggered by any result. Real-data results are development
  evidence obtained on validation data that was also used for MLP checkpoint selection, so
  they favour the MLP selection rule and are labelled as such.

## 8. Runtime gate

Per-fit cost is measured on one throwaway probe run per case (`max_epochs = 1`, a scratch
directory outside `outputs/`) before the nine runs start. If the extrapolated cost of the
declared plan is impractical for this session, execution stops and the measured costs are
reported **before** any budget change; the budget is not reduced silently.

## 9. Artifacts

- New: `configs/sp500_w1000_ingest.json`, `data/sp500_w1000/` (panel + audit, gitignored),
  `configs/experiment-20261104/*.json`, nine run directories, this protocol.
- Preserved: `data/sp500/`, every existing `outputs/*` run, all sources.
- `STATUS.md` / `DECISIONS.md`: one entry each after execution.
