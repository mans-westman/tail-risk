# Tailrisk — conditional tail-risk research toolkit

A research toolkit for students and quantitative researchers comparing next-session value-at-risk (VaR) and expected shortfall (ES) models on one daily return series. It brings data timing, common-sample scoring, failure accounting and reproducibility evidence into one workflow. Synthetic cases supply known conditional risk; real data support forecast evaluation without an oracle. Returns are signed decimal log returns and `alpha` is a lower-tail probability; the supported scoring domain is `es <= var < 0`.

**Maturity: working research prototype.** Historical simulation, a fitted GARCH–Student-t baseline and an alpha-conditioned joint MLP all produce forecasts, and the CLI can independently verify, score, and report them. All three models run on synthetic data and on an ingested real-data panel; the joint MLP has so far been run on the train/validation periods only (2016–2019 is development evidence, not a held-out result), its saved forecasts are replayed by `verify` from the checkpoint rather than refitted, and the available comparisons do not establish predictive superiority. Development status and dated evidence are kept in a local working log (`STATUS.md`), which is not distributed with the public package; this paragraph is the public-facing maturity statement and should stay short.

## Approved delivery direction

The next milestone adds a documented returns-CSV input, a small public Python model-adapter interface, and a comparison report that separates artifact integrity, verification coverage and statistical results. Humans and a regular WorkSwarm skill will use the same CLI. These additions are **planned, not yet implemented**; current commands below remain the working entry points.

The [Stage 2 implementation roadmap](docs/STAGE2_IMPLEMENTATION.md) fixes the contracts and ordered acceptance gates. The toolkit does not certify absence of all leakage, regulatory suitability or universal model superiority. A novel constrained model is an optional later research extension, not a dependency of the reusable workflow.

## Repository map

```text
tailrisk/                    # run commands here; Git root
  README.md                  # this file: purpose, setup, commands, artifact meaning
  docs/                      # approved implementation roadmap; public usage docs added by slice
  configs/                   # JSON run and ingest configurations
  src/tailrisk/              # package and CLI
  tests/                     # numerical, timing, mask, and diagnostic checks
  outputs/<run_id>/          # generated runs (ignored except its README)
```

Local working documents (`AGENTS.md`, `STATUS.md`, `DECISIONS.md`, `PROJECT_GUIDE.md`, and `LITERATURE_REVIEW.md`) are gitignored and are not required to follow this public quick start.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` declares numpy, scipy and pandas as runtime dependencies, pytest plus `arch` for development, and an optional `neural` extra for the alpha-conditioned MLP core. Install it with `python -m pip install -e '.[dev,neural]'` when running or developing the MLP: conventional-only CLI runs do not require JAX or Optax.

The joint MLP is three layers: a mathematical core (`tailrisk.models.joint_mlp`: initialization, forward pass, FZ0 scores and their mean), a trainer (`tailrisk.models.joint_mlp_training`: train-only feature standardization, clipped-Adam fitting with validation-based checkpoint selection, original-unit prediction, NPZ+JSON inference checkpoint) and an adapter (`tailrisk.models.joint_mlp_model`) that builds the six causal features and per-origin return scale from one split's returns, reports unusable origins as explicit failed rows instead of dropping them, and turns the selected checkpoint into `forecast.csv` rows. The trainer consumes prepared arrays and decides no calendar cutoffs itself; the adapter receives only the origins the run's protocol permits. Installing the extra is required only when `models` includes `joint_mlp`. Dependency ranges are not a locked environment. Each run records its interpreter and dependency versions in `manifest.json`.

## Commands

```sh
python -m pytest -q
python -m tailrisk run      --config configs/smoke.json --output outputs/my-run
python -m tailrisk run      --config configs/smoke_mlp.json --output outputs/my-mlp-run   # + joint MLP
python -m tailrisk verify   --run outputs/my-run
python -m tailrisk evaluate --run outputs/my-run
python -m tailrisk report   --run outputs/my-run

# real development: acquire once, then forecast train/validation only
python -m tailrisk ingest   --config configs/sp500.json --out data/sp500
python -m tailrisk run      --config configs/sp500_dev.json --output outputs/sp500-dev
python -m tailrisk verify   --run outputs/sp500-dev
python -m tailrisk evaluate --run outputs/sp500-dev
python -m tailrisk report   --run outputs/sp500-dev
```

`run` consumes either a declared synthetic DGP (`configs/smoke.json`) or an ingested panel (`configs/sp500_dev.json`, `data.kind: "returns_panel"`), builds causal features, and writes forecasts. A synthetic config declares `dgp`, `partitions` and `seeds`; a panel config declares `data.panel` instead, and supplying both is an error rather than a silent preference for one of them. `verify` re-derives the saved artifacts independently (including deterministic replay) and writes `artifact_verification.json`; it does not modify `checks.json`. `evaluate` scores the saved forecasts and writes `metrics.csv`. `report` assembles `report.md` from existing artifacts only.

Use a new or empty output directory for every run. `run` refuses an existing file or nonempty directory before loading data or fitting models. Verification, evaluation and reporting explicitly update their own derived files within the selected run.

`ingest` is the only command that touches the network; `run` never downloads, it reads the panel `ingest` wrote, so a run is reproducible from local files alone. It obtains one dated snapshot (`^GSPC` price-index closes by default), validates it, and writes `panel.csv` plus `data_audit.json`. Add `--from-raw` to re-derive from a saved payload without downloading, `--reference` to cross-check a return file whose units are declared (never inferred), and `--keep-raw` to store the payload, its SHA-256, and retrieval time. Market data is not redistributed: `data/` is ignored, so acquisition is reproducible from the config rather than from a committed cache.

The panel is one causal table per split. Splits are assigned by **target date**, and validation/test keep trailing returns from the preceding period, so their opening sessions still have a full feature and historical window instead of being cropped. A split that cannot be given that context fails loudly. `is_scored` marks rows whose target lies in the split and whose origin has the required history; `prior_returns` and `origin_position` make the warm-up auditable, and `spans_long_gap` flags returns spanning a closure.

## Real-data protocol

Development uses `configs/sp500_dev.json`, which executes train and validation only. The earlier conventional-model run already evaluated the 2020–2025 test period; it is an exposed retrospective comparison, not an untouched final holdout. Keep `configs/sp500_run.json` for explicitly planned full-period evaluation, not routine development.

Panel runs require `train_end` and `validation_end` as `YYYY-MM-DD` strings, directly or through the ingestion audit. The resolved run configuration freezes both dates; verification uses those frozen dates and checks the consumed panel hash.

A panel run is frozen rather than tuned: the split cutoffs, historical window, alphas and model set come from the config, and `run` refuses a panel that cannot supply every scored origin with a full trailing window. Every split carries its own causal context, so no opening session is discarded. Forecast rows carry real session dates instead of step indices, and only origins the panel marked `is_scored` are scored.

Legacy development configs use 100 returns; at alpha <= 0.01 their empirical VaR and ES both equal the window minimum. The subsequent frozen 2026-11-04 comparison adopted W=1000 before running, on tail-resolution and sample-availability grounds. Its configs are in `configs/experiment-20261104/`. New Stage 2 examples will specify W=1000 explicitly; old configs and results are preserved, not silently changed.

The GARCH-t likelihood is fitted exactly once, on the train split, and frozen for every split a run executes — validation always, test only in an explicitly requested full-period run; the parameters are recorded in `manifest.json`. `verify` proves this independently by refitting from the panel's train slice alone and replaying every saved forecast, and it names any dependency drift between the verifying interpreter and the run's manifest, since a refit is only comparable across matching numeric builds.

A market-data run has no ground-truth oracle, so it writes no `oracle_curves.csv`, `verify` reports oracle comparisons as *not applicable* rather than passed, and `evaluate` leaves the oracle columns undefined. A stray oracle file in a real-data run fails verification. Because the cutoffs and the estimator were fixed before those partitions were scored, changing them later is a new specification, not a rerun.

## What the artifacts mean

- `forecasts.csv` — one row per model, origin date, and alpha: `var`, `es`, `scale`, `realized_return`, `status`, in original return units.
- `oracle_curves.csv` — evaluation-only synthetic truth (latent sigma and analytical oracle VaR/ES). Never a model input, and written only for synthetic runs: a real panel has no oracle at all.
- `checkpoints/joint_mlp/` — written only when `joint_mlp` is configured: `checkpoint.json` (architecture, feature order, normalization, alpha grid, training log, selected epoch) and `arrays.npz` (parameters). `manifest.json` records the SHA-256 of both files plus the eligible row counts and selected epoch. `verify` loads this checkpoint, recomputes forecasts from the causal inputs and compares them with `forecasts.csv`; it never retrains, and a replaced or missing file fails verification. The smoke budget in `configs/smoke_mlp.json` is deliberately small: it is an execution check, not a trained result.
- `failures.csv` — one row per failed forecast (`series_id, model_id, origin_date, target_date, alpha, scale, reason`), including runs with no failures (header only). Failed rows also stay in `forecasts.csv` with `status: "failed"` and no value, so the forecast schema is unchanged. `verify` requires the documented keys to match the failed rows one-to-one and recomputes each origin's six features to check that `ok` and `failed` are assigned exactly where the inputs allow, and that the stated reason is the one those features imply. Required only for `joint_mlp` runs. `verify` also derives the expected forecast keys per split, origin and alpha level from the permitted origins, the 63-return warm-up, the executed splits and the alpha grid, so a dropped origin, alpha level or whole split fails verification instead of shrinking the run silently.
- `metrics.csv` — per split/model/alpha scores on a **common scoring mask**: within each (series, alpha) cell every model is scored on the same origins, so `n` is identical across models and means are comparable. `n_available`, `n_failed_rows` and `n_dropped_by_mask` expose the coverage cost; `n_invalid_output` counts non-finite predictions and violations of `es <= var < 0`, which are flagged rather than clipped. A model that fails outright empties its cell (all models report `n = 0`) instead of letting the survivors be scored alone, and a model listed in the resolved config that wrote no rows at all fails `evaluate` rather than shrinking the comparison. Scoring also rejects duplicate `(series, model, alpha, origin)` keys and target dates that do not follow their origins. Backtests are Kupiec (unconditional coverage) and Christoffersen (independence); `nan` means the statistic was undefined (for example no exceedances at all) and is reported as undefined rather than imputed. `n_transitions` and `n_chain_breaks` record how many chronologically adjacent pairs entered the independence test and how many were severed by a scoring-mask or session gap, so a broken chain is visible instead of silently joining non-adjacent days.
- `checks.json` — offline mathematical checks; `manifest.json` / `config.resolved.json` — provenance; `summary.md`, `report.md` — readable outputs.

Synthetic-only runs are execution checks, not statistical evidence. Oracle comparisons are implementation checks, not forecasting skill. A failed row (`status: "failed"`) carries a real origin and target, an explicit reason in `failures.csv` (checked by `verify` against the rebuilt features) and no forecast value: it is counted in `n_failed_rows` and never imputed, so coverage loss stays visible instead of shrinking the sample quietly.

## Reuse and limits

This is a course project intended to become a reusable [openJiuwen WorkSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) workflow; fresh-run skill evidence is not yet collected. Substantial AI assistance has been used for planning and implementation; reviewed code and saved execution evidence determine what is established.

No project license has been selected yet, and third-party attribution must be resolved before public release. Neither these forecasts nor this prototype establish suitability for trading or regulatory use.
