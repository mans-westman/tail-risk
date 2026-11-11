---
name: tail-risk-audit
description: Run and audit the tailrisk VaR/ES comparison workflow on one daily return series. Use when the user wants to compare next-session value-at-risk and expected shortfall models, run or verify a tailrisk run directory, ingest a declared returns CSV, score saved forecasts, render the evidence report, or explain what a tailrisk run does and does not establish.
---

# Tail-risk audit

Drive the `tailrisk` CLI end to end on one return series and report what the evidence
actually supports. You run the same commands a human would; no model is queried during
numerical execution, and you never edit a run's artifacts by hand.

## Preconditions

**Interpreter.** Use the project's own environment, `.venv/bin/python` at the repository root, or
another interpreter that already has the package. A bare `python3` on this machine is an old
system interpreter without NumPy, and every command below fails there. Confirm before starting:
`.venv/bin/python -m tailrisk --help` must print usage. Throughout this skill, `python` means that
interpreter.

**Extras.** The quickstart needs `baselines` (`arch`); add `plots` for figures, and `neural` (plus
`JAX_ENABLE_X64=1`) only for the joint MLP:
`python -m pip install -e '.[baselines]'`.

**Working directory.** Run from the repository root, so `python -m tailrisk` resolves and, for
custom adapters, so factory ids such as `examples.custom_model.model:create_model` import.

**Output directory.** Choose a **new or empty** output directory. `run` refuses a nonempty one;
that refusal is a feature, not an error to work around.

If `python -m tailrisk --help` fails, stop and report the installation problem. Do not edit the
package to make a run proceed.

## Workflow

### 1. Inspect the inputs before running anything

| Input | Check | If it is wrong |
|---|---|---|
| A config in `configs/` or `configs/examples/` | `schema_version: 1`, `models`, `alpha_grid`, `historical_window`, and either `dgp`+`partitions`+`seeds` or `data.panel` | report the missing key; do not add a second data source |
| A returns CSV (only with `ingest`) | exactly `session_date,return`, ISO dates, strictly ascending, unique, finite, decimal log units | report the offending rows; never sort or dedupe the file |
| Cutoffs | `train_end` / `validation_end` present for panel runs, and the split you intend to *report* is the split actually executed | state which splits the run will execute |

Read the config out loud to the user in one short block (models, window, alphas, splits,
seeds). A run whose splits are misread produces confident, wrong evidence.

### 2. Resolve the configuration

Pick one of:

- **synthetic example** — `configs/examples/student_t.json` (primary), `gaussian.json`,
  `student_t_mlp.json`;
- **declared CSV** — a returned file plus an ingest config, then
  `configs/examples/student_t.json` is *not* the right precedent; copy
  `examples/custom_model/run.json` instead;
- **real panel** — `configs/sp500_dev.json` after `ingest`, remembering that its test period is
  already an exposed retrospective comparison.

State the run's question and its splits before executing. If the user's question needs a test
split that the config does not declare, say so instead of quietly enabling one.

### 3. Run

```sh
python -m tailrisk run --config <config> --output <fresh-dir>
```

Confirm from the printed summary: partition sizes, forecast row count per model, and
`offline checks: N/N passed`. A partial or failed run must not be presented as a comparison.

### 4. Verify

```sh
python -m tailrisk verify --run <dir>
```

Read the summary line: `X/Y applicable checks passed; N not applicable; U unavailable`.
`unavailable` means coverage is partial — say which capability is missing (typically no
replay for a model that declares `replay_mode="none"`). Never translate that into "verified
clean". A syntactic failure exits nonzero: report it, do not patch the artifacts.

### 5. Score

```sh
python -m tailrisk evaluate --run <dir>
```

Confirm the four tables were written and that `n` is identical across models within a cell.
Differences between models are only comparable on that common mask; if `n_dropped_by_mask`
is large, say what coverage was lost.

### 6. Report

```sh
python -m tailrisk report --run <dir>          # add --plots if Matplotlib is installed
```

Then summarise for the user in their terms: the question, the splits, the models and their
conditions, the loss differences with `n`, exceedance counts and undefined tests, integrity
and coverage findings, and the limits. Quote numbers from the tables, not from memory.

### 7. Explain evidence and failures

- A failed forecast carries a reason in `failures.csv`; the reason is the model's own
  account, checked against rebuilt features where the model is a built-in.
- A `nan` backtest statistic means undefined (for example, no exceedances), not "passed".
- An invalid output (`n_invalid_output > 0`) means a forecast broke `es <= var < 0`; it is
  counted and not repaired.

## Guardrails

- Never claim model superiority, statistical significance, regulatory suitability, or the
  absence of leakage. Comparisons are whole-method; differences have no interval.
- Never call a synthetic run evidence about markets, and never call oracle accuracy
  forecasting skill.
- Do not stage, commit, push or publish anything as part of this workflow unless the user
  explicitly asks in that turn.
- Do not install packages or download data silently. `ingest` is the only command that touches
  the network, and it needs the user's consent.

## References

Read only what the current step needs.

| File | Read it when |
|---|---|
| `references/METHODS.md` | you must explain a formula, an estimator, attribution, or a limitation |
| `references/DATA_CONTRACT.md` | the user supplies a returns CSV or an ingestion config fails |
| `references/MODEL_ADAPTER.md` | the user wants their own model, or a factory/checkpoint error appears |
| `references/REPORTING.md` | you must explain a column, a report section or a figure |
| `references/ENVIRONMENT.md` | the user asks about versions, extras, install failures or runtimes |

These are byte-identical copies of the repository's `docs/` pages. Enforce that with
`python skills/tail-risk-audit/check_references.py` from the repository root: update the
canonical page, then copy; do not edit a reference in place. Run from an installed copy, the
script reports that there is no `docs/` to compare against and exits 0.
