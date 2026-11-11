# tailrisk — conditional tail-risk research toolkit

A research toolkit for comparing next-session value-at-risk (VaR) and expected shortfall
(ES) models on **one daily return series**. It supplies common timing, a common scoring
sample, failure accounting, provenance and an inspectable report, so a comparison can be
reproduced and audited instead of assembled in a notebook.

Returns are signed decimal log returns and `alpha` is a lower-tail probability, so losses are
negative and the supported scoring domain is `es <= var < 0`.

**Maturity: working research prototype.** Historical simulation, a fitted GARCH(1,1)-Student-t
baseline and an optional alpha-conditioned joint MLP all produce forecasts, and the CLI
verifies, scores and reports them independently. The MLP has so far been run on the
train/validation periods only, so its saved forecasts are replay evidence, not a held-out
result. No comparison in this repository establishes predictive superiority.

## Install

Obtain the repository, then run every command on this page **from its root**:

```sh
git clone https://github.com/mans-westman/tail-risk.git
cd tail-risk
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[baselines]'
```

The working directory matters. `configs/`, `docs/`, `examples/` and `skills/` live in the
checkout, and the worked custom adapter is imported as `examples.custom_model.model`, which
resolves only from the root. Commands are written `python -m tailrisk ...`; after
`source .venv/bin/activate` the installed `tailrisk` script is equivalent for the synthetic and
panel runs, but not for a run whose config names a factory from this checkout.

`baselines` adds `arch`, which the GARCH-t model needs. Everything else is optional and
genuinely so: without `neural` (JAX, Optax) the MLP is simply unavailable, without `plots`
(Matplotlib) `report` still writes Markdown and JSON, and without `data` the exchange-calendar
cross-check in `ingest` is recorded as skipped. Depth: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Quickstart, offline

```sh
python -m tailrisk run      --config configs/examples/student_t.json --output outputs/my-run
python -m tailrisk verify   --run outputs/my-run
python -m tailrisk evaluate --run outputs/my-run
python -m tailrisk report   --run outputs/my-run
```

That simulates a GARCH(1,1) Student-t(5) series, forecasts it with historical simulation and
GARCH-t, verifies the artifacts, scores them on a common mask, and renders a Markdown/JSON
report. No network, no market data, and no test split is declared. Add `--plots` to the last
command (plus `'.[plots]'`) for three PNGs. `configs/examples/gaussian.json` is the same
experiment with Gaussian innovations; `configs/examples/student_t_mlp.json` adds the MLP.

Every example is an execution and reuse check with limited tail counts, not performance
evidence. [examples/README.md](examples/README.md) walks through all of them, including a
custom adapter on a locally generated CSV.

## Commands

| Command | Does | Writes |
|---|---|---|
| `run --config <json> --output <dir>` | simulates or reads the configured data, fits each model once, emits one forecast per model/origin/alpha | `forecasts.csv`, `failures.csv`, `oracle_curves.csv` (synthetic only), `manifest.json`, `config.resolved.json`, `checks.json`, `summary.md` |
| `verify --run <dir>` | re-derives artifacts from the inputs; refits GARCH-t and replays the MLP checkpoint where applicable | `artifact_verification.json`, `verification_coverage.json` |
| `evaluate --run <dir>` | scores saved forecasts on one common mask | `metrics.csv`, `score_decomposition.csv`, `paired_scores.csv`, `curve_diagnostics.csv` |
| `report --run <dir> [--plots]` | renders saved artifacts only; no fit, no download | `report.md`, `report.json`, `plots/*.png` |
| `ingest --config <json> --out <dir>` | the only command that touches the network: one dated snapshot, validated | `panel.csv`, `data_audit.json` |

`run` refuses an existing file or nonempty output directory before loading data or fitting, so
use a fresh path for each run. `verify`, `evaluate` and `report` update only their own derived
files inside the chosen run.

## What the artifacts mean

- **`forecasts.csv`** — one row per model, origin and alpha: `var`, `es`, `scale`,
  `realized_return`, `status`, in original return units.
- **`failures.csv`** — one row per **affected key**, covering two different cases with exactly
  one explanation each: an explicit model failure (`status: "failed"`, no value) and an
  invalid output — a finite-shaped answer that is non-finite or violates `es <= var < 0` and
  therefore stays in `forecasts.csv` for the scorer to count. Failed rows keep their place in
  `forecasts.csv` with no value; nothing is imputed. The provenance of the reason differs by
  case: an explicit failure carries the **model's own declared** reason, which for a custom
  adapter is author-reported and not independently established, while an invalid output
  carries the runner's recomputed vocabulary (`non_finite_output`, `outside_supported_domain`)
  that `verify` recomputes and demands an exact match. For the built-in MLP, `verify`
  additionally rebuilds the six features at each origin to check that `ok` and `failed` were
  assigned where the inputs allow.
- **`metrics.csv`** — per split/model/alpha scores on a **common mask**: within each
  `(series, alpha)` cell every model is scored on the same origins, so `n` is identical and
  means are comparable. `n_available`, `n_failed_rows`, `n_dropped_by_mask` and
  `n_invalid_output` expose the coverage cost. Backtests are Kupiec and Christoffersen; `nan`
  means undefined, not zero.
- **`score_decomposition.csv`**, **`paired_scores.csv`**, **`curve_diagnostics.csv`** — the
  additive parts of the loss, paired mean differences on the same mask, and within-alpha
  admissibility plus adjacent-alpha crossings. Semantics in [docs/REPORTING.md](docs/REPORTING.md).
- **`artifact_verification.json`**, **`verification_coverage.json`** — every integrity check
  with `passed`/`failed`/`unavailable`/`not_applicable` and a reason. Unavailable replay is
  partial coverage, never a pass.
- **`manifest.json`, `config.resolved.json`, `checks.json`** — provenance: frozen
  configuration, interpreter and dependency versions, fitted parameters, model capabilities,
  factory hashes, and the offline mathematical checks.

## Bring your own data

An ingestion config can read a **declared local returns CSV** (`session_date,return`, ISO
dates, decimal log returns, ascending and unique) instead of downloading. Local CSV ingestion
is entirely **offline**; only the provider path — `ingest` against a market-data source such as
`configs/sp500.json` — uses the network, and it is the only command in the toolkit that does.
Nothing is sorted, inferred or repaired: an unresolved gap stops ingestion. The audit records
the source hash and the declared cutoffs, and a run then reads only local files. Contract:
[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

A real panel run needs `train_end` and `validation_end`; the resolved run freezes them, and
`verify` re-checks the consumed panel hash. `configs/sp500_dev.json` executes train and
validation only. The 2020–2025 test period of the earlier conventional-model run is already an
exposed retrospective comparison, not an untouched holdout, so `configs/sp500_run.json` is for
explicitly planned full-period evaluation rather than routine development. Market data is not
redistributed: `data/` is ignored and `ingest` re-acquires from the config.

## Bring your own model

A model is a small Python adapter — `fit`, `predict_one`, `summary`, plus a declared
`ModelSpec` — named in the configuration as `module:function`. The runner owns IDs, targets and
realized returns, and every prediction argument contains a **bounded causal prefix**: a history
array that ends at the origin, owns its memory, and carries no target, oracle, latent
volatility path or later row. Checkpoints are optional and replayed by `verify` with `fit`
disabled. No evaluator edit is needed to register a model.

That boundary is what the toolkit hands over, not a sandbox. A factory is arbitrary Python
executing in your process: imported code can still read files, reach the network or hold
pretrained state, and the manifest records its declarations plus a hash of the imported module
rather than proving their absence. There is **no universal leakage guarantee**, and the
verification checks listed in [docs/METHODS.md](docs/METHODS.md) are the ones that ran.

Contract and worked example: [docs/MODEL_ADAPTER.md](docs/MODEL_ADAPTER.md).
`examples/bring_your_model/` holds a standalone EWMA–Gaussian forecast that is deliberately
*not* registered or integrated — the raw function a researcher adapts through that contract.

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `baselines` | `arch` | `garch_student_t` |
| `neural` | `jax`, `optax` | `joint_mlp` (also needs `JAX_ENABLE_X64=1`) |
| `plots` | `matplotlib` | `report --plots` |
| `data` | `pandas_market_calendars` | the optional exchange-session cross-check in `ingest` |
| `dev` | `pytest`, `arch` | the test suite |

## WorkSwarm skill

`skills/tail-risk-audit/` is a regular [openJiuwen WorkSwarm](https://github.com/openJiuwen-ai/jiuwenswarm)
skill: one CLI workflow (inspect inputs → resolve config → run → verify → evaluate → report →
explain evidence and failures) with the public pages copied under `references/` so a copied
skill does not depend on this checkout. It runs the same commands as a human; no model is called
inside the numerical execution, and `skills/tail-risk-audit/check_references.py` keeps the
copies in step with `docs/`.

To install it, copy the directory into the workspace skills root and start a new session so the
runtime rescans that directory:

```sh
cp -R skills/tail-risk-audit ~/.jiuwenswarm/agent/workspace/skills/
# then start a new WorkSwarm session and ask it to run a tail-risk audit
```

The engine still has to be installed where that session runs (`python -m pip install -e
'.[baselines]'`, or the package plus its extras); the skill does not vendor the code.

**Where the evidence stops.** The package passes the runtime's own skill validator, and its
scanner and inventory code parse it into a registered skill with a name, description and content
hash. What has **not** been observed is live installation followed by a session reload, and a
fresh agent executing the workflow in a new conversation. Those are the remaining gate, so the
current claim is "loadable, and the workflow is documented and tested by hand" — not "a fresh
agent has run it".

## What this does not establish

- No leakage certificate. The verification checks are the ones listed in
  [docs/METHODS.md](docs/METHODS.md); trusted local plugin code is not sandboxed, and no check
  detects contaminated pretraining or researcher selection.
- No ranking, significance or suitability claim: comparisons are whole-method, differences
  carry no confidence interval, and repeated seeds on one series are not independent evidence.
- No ES-specific backtest and no bootstrap inference; both are unavailable, not passed.
- Synthetic runs are implementation and execution checks, not evidence about markets.

## Licence

No project licence has been selected. Choosing one is a prerequisite for publication and is
deliberately left to the project owner; attribution for methods and dependencies is in
[docs/METHODS.md](docs/METHODS.md).

Substantial AI assistance was used for planning and implementation. Reviewed code and saved
execution evidence, not the assistance, determine what is established.
