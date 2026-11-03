# Tailrisk

Next-session value-at-risk (VaR) and expected shortfall (ES) forecasting for one daily return series, with synthetic cases where the true conditional risk is known. Returns are signed decimal log returns; `alpha` is a lower-tail probability, so forecasts satisfy `es <= var < 0` in normal cases.

**Maturity: working research prototype.** Historical simulation and a fitted GARCH–Student-t baseline produce forecasts, and the CLI can independently verify, score, and report them. Both models run on synthetic data and an ingested real-data panel. The neural baseline is not implemented, and the available comparisons do not establish predictive superiority. Development status and dated evidence are kept in a local working log (`STATUS.md`), which is not distributed with the public package; this paragraph is the public-facing maturity statement and should stay short.

## Repository map

```text
tailrisk/                    # run commands here; Git root
  README.md                  # this file: purpose, setup, commands, artifact meaning
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

`pyproject.toml` declares numpy, scipy and pandas as runtime dependencies and pytest plus `arch` as development dependencies. JAX/Optax are installed in the checked environment but are not yet required by working source; they will be declared when the neural baseline lands. Dependency ranges are not a locked environment. Each run records its interpreter and dependency versions in `manifest.json`.

## Commands

```sh
python -m pytest -q
python -m tailrisk run      --config configs/smoke.json --output outputs/my-run
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

A panel run is frozen rather than tuned: the split cutoffs, the 100-session window, the six alphas and the model set come from the config, and `run` refuses a panel that cannot supply every scored origin with a full trailing window. Every split carries its own causal context, so no opening session is discarded. Forecast rows carry real session dates instead of step indices, and only origins the panel marked `is_scored` are scored.

The development historical window is 100 returns. At alpha <= 0.01, its empirical VaR and ES both equal the window minimum; these estimates cannot distinguish the quantile from the average of a more finely resolved tail. Window choice for the research comparison remains to be settled without using test-period performance.

The GARCH-t likelihood is fitted exactly once, on the train split, and frozen for every split a run executes — validation always, test only in an explicitly requested full-period run; the parameters are recorded in `manifest.json`. `verify` proves this independently by refitting from the panel's train slice alone and replaying every saved forecast, and it names any dependency drift between the verifying interpreter and the run's manifest, since a refit is only comparable across matching numeric builds.

A market-data run has no ground-truth oracle, so it writes no `oracle_curves.csv`, `verify` reports oracle comparisons as *not applicable* rather than passed, and `evaluate` leaves the oracle columns undefined. A stray oracle file in a real-data run fails verification. Because the cutoffs and the estimator were fixed before those partitions were scored, changing them later is a new specification, not a rerun.

## What the artifacts mean

- `forecasts.csv` — one row per model, origin date, and alpha: `var`, `es`, `scale`, `realized_return`, `status`, in original return units.
- `oracle_curves.csv` — evaluation-only synthetic truth (latent sigma and analytical oracle VaR/ES). Never a model input, and written only for synthetic runs: a real panel has no oracle at all.
- `metrics.csv` — per split/model/alpha scores on a **common scoring mask**: within each (series, alpha) cell every model is scored on the same origins, so `n` is identical across models and means are comparable. `n_available`, `n_failed_rows` and `n_dropped_by_mask` expose the coverage cost; `n_invalid_output` counts non-finite predictions and violations of `es <= var < 0`, which are flagged rather than clipped. A model that fails outright empties its cell (all models report `n = 0`) instead of letting the survivors be scored alone, and a model listed in the resolved config that wrote no rows at all fails `evaluate` rather than shrinking the comparison. Scoring also rejects duplicate `(series, model, alpha, origin)` keys and target dates that do not follow their origins. Backtests are Kupiec (unconditional coverage) and Christoffersen (independence); `nan` means the statistic was undefined (for example no exceedances at all) and is reported as undefined rather than imputed. `n_transitions` and `n_chain_breaks` record how many chronologically adjacent pairs entered the independence test and how many were severed by a scoring-mask or session gap, so a broken chain is visible instead of silently joining non-adjacent days.
- `checks.json` — offline mathematical checks; `manifest.json` / `config.resolved.json` — provenance; `summary.md`, `report.md` — readable outputs.

Synthetic-only runs are execution checks, not statistical evidence. Oracle comparisons are implementation checks, not forecasting skill.

## Reuse and limits

This is a course project intended to become a reusable [openJiuwen WorkSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) workflow; fresh-run skill evidence is not yet collected. Substantial AI assistance has been used for planning and implementation; reviewed code and saved execution evidence determine what is established.

No project license has been selected yet, and third-party attribution must be resolved before public release. Neither these forecasts nor this prototype establish suitability for trading or regulatory use.
