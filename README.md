# Tailrisk

Next-session value-at-risk (VaR) and expected shortfall (ES) forecasting for one daily return series, with synthetic cases where the true conditional risk is known. Returns are signed decimal log returns; `alpha` is a lower-tail probability, so forecasts satisfy `es <= var < 0` in normal cases.

**Maturity: working research prototype.** Historical simulation and a fitted GARCH–Student-t baseline produce forecasts, and the CLI can independently verify, score, and report them. A neural baseline and a real-data evaluation case are not implemented yet, so no predictive comparison or superiority claim exists anywhere in this package. Development status and dated evidence are kept in a local working log (`STATUS.md`), which is not distributed with the public package; this paragraph is the public-facing maturity statement and should stay short.

## Repository map

```text
tailrisk/                    # run commands here; Git root
  README.md                  # this file: purpose, setup, commands, artifact meaning
  STATUS.md                  # current capabilities, dated evidence, limitations, next actions
  DECISIONS.md               # append-only log of consequential choices
  PROJECT_GUIDE.md           # scientific specification and staged plan (stable reference)
  AGENTS.md                  # working rules, verification standards, Git guidance
  LITERATURE_REVIEW.md       # bounded literature/novelty audit
  configs/                   # JSON run configurations
  src/tailrisk/              # package and CLI
  tests/                     # numerical, timing, mask, and diagnostic checks
  outputs/<run_id>/          # generated runs (ignored except its README)
```

Four documents, four jobs: `STATUS.md` is the living log (capabilities, dated evidence, limitations), `DECISIONS.md` only grows (append-only rationale), `PROJECT_GUIDE.md` is the frozen specification, and this `README.md` describes usage and should change rarely. `AGENTS.md` holds working rules for implementers. If a capability claim needs updating, change it in `STATUS.md` and leave this file alone.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` declares numpy, scipy and pandas as runtime dependencies and pytest plus `arch` as development dependencies. JAX/Optax are installed in the checked environment but are not yet required by working source; they will be declared when the neural baseline lands. Ranges are not a locked environment — see `STATUS.md` for the versions actually tested.

## Commands

```sh
python -m pytest -q
python -m tailrisk run      --config configs/smoke.json --output outputs/my-run
python -m tailrisk verify   --run outputs/my-run
python -m tailrisk evaluate --run outputs/my-run
python -m tailrisk report   --run outputs/my-run
```

`run` simulates, builds causal features, and writes forecasts. `verify` re-derives the saved artifacts independently (including deterministic replay) and writes `artifact_verification.json`; it does not modify `checks.json`. `evaluate` scores the saved forecasts and writes `metrics.csv`. `report` assembles `report.md` from existing artifacts only.

Use a new output directory for every run; the CLI does not protect existing outputs from being overwritten.

## What the artifacts mean

- `forecasts.csv` — one row per model, origin date, and alpha: `var`, `es`, `scale`, `realized_return`, `status`, in original return units.
- `oracle_curves.csv` — evaluation-only synthetic truth (latent sigma and analytical oracle VaR/ES). Never a model input.
- `metrics.csv` — per split/model/alpha scores on a **common scoring mask**: within each (series, alpha) cell every model is scored on the same origins, so `n` is identical across models and means are comparable. `n_available`, `n_failed_rows` and `n_dropped_by_mask` expose the coverage cost; `n_invalid_output` counts inadmissible `es >= 0` rows, which are flagged rather than clipped. A model that fails outright empties its cell (all models report `n = 0`) instead of letting the survivors be scored alone, and a model listed in the resolved config that wrote no rows at all fails `evaluate` rather than shrinking the comparison. Scoring also rejects duplicate `(series, model, alpha, origin)` keys and target dates that do not follow their origins. Backtests are Kupiec (unconditional coverage) and Christoffersen (independence); `nan` means the statistic was undefined (for example no exceedances at all) and is reported as undefined rather than imputed. `n_transitions` and `n_chain_breaks` record how many chronologically adjacent pairs entered the independence test and how many were severed by a scoring-mask or session gap, so a broken chain is visible instead of silently joining non-adjacent days.
- `checks.json` — offline mathematical checks; `manifest.json` / `config.resolved.json` — provenance; `summary.md`, `report.md` — readable outputs.

Synthetic-only runs are execution checks, not statistical evidence. Oracle comparisons are implementation checks, not forecasting skill.

## Reuse and limits

This is a course project intended to become a reusable [openJiuwen WorkSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) workflow; fresh-run skill evidence is not yet collected. Substantial AI assistance has been used for planning and implementation; reviewed code and saved execution evidence determine what is established.

No project license has been selected yet, and third-party attribution must be resolved before public release. Neither these forecasts nor this prototype establish suitability for trading or regulatory use.
