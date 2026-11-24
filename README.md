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

**Release status.** Source version `0.1.0`, licensed under the [MIT License](LICENSE); no tag
or package index publication yet. The checkout is the distribution: installation is
`pip install -e .` from the cloned tree.

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

### Reproducing the tested environment

`pyproject.toml` declares ranges, for library users and for future platforms. The **exact**
environment in which this release was tested is pinned in
[`requirements-tested.txt`](requirements-tested.txt) (core, `baselines`, `plots`, `dev` and the
`neural` extras, with their transitive dependencies):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
```

`--no-deps` is deliberate: the requirements file is already the complete pinned set, and letting
the package resolve its ranges on top could move a version away from the tested one.
`requires-python = ">=3.10"` is a **declared** compatibility range whose dependency requirements
resolve on 3.10; the tested interpreter is CPython 3.13.15, and no other version is claimed as
tested. The record, the measured runtimes and the optional-extra behaviour are in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

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
command (plus `'.[plots]'`) for three PNGs. On a machine like the one in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) expect 41,232 forecast rows, `verify` reporting
27/27 applicable checks with 0 unavailable, 24 metric rows, and three figures; a run takes
about 20 seconds.

`configs/examples/gaussian.json` is the same experiment with Gaussian innovations;
`configs/examples/gaussian_ewma.json` adds the researcher-supplied EWMA model (below);
`configs/examples/student_t_mlp.json` adds the MLP.

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
*not* registered — the raw function a researcher adapts through that contract.
`examples/ewma_gaussian_adapter/` is the integration of exactly that function: it calls the
supplied code with the causal history the runner hands over and maps the result onto contract
v1, without editing or reimplementing the mathematics. The S2.6 trial below is that case run
end to end by a fresh agent.

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

## WorkSwarm trial (S2.6)

`skills/tail-risk-audit/` was installed into a live workspace and a **fresh WorkSwarm agent**
was given the repository and the task, not the solution. The agent read the skill and the
adapter documentation, wrote `examples/ewma_gaussian_adapter/` and
`configs/examples/gaussian_ewma.json`, and ran both documented runs end to end.

| What ran | Result |
|---|---|
| documented Gaussian example | 41,232 forecast rows; `verify` 27/27 applicable, 0 unavailable; 0 failed rows, 0 invalid outputs |
| comparison with the supplied EWMA model | 67,476 rows (adapter: 26,244); `verify` 33/33 applicable; adapter checkpoint replays 26,244/26,244 with 0 mismatches |
| baseline invariance | the 24 baseline metric rows are equal across the two runs (`pandas.DataFrame.equals` on the saved `metrics.csv` tables — an equality check on the parsed rows, not a byte-level file comparison) |
| scoring | common mask: 1,999 train / 499 validation origins; EWMA 12 exceedances on the matched sample at 0.025 (43 on its own 1,437 available origins) |

The supplied model file was read and called, never edited. Record, commands, runtimes,
figures and the limitations of the evidence: [evidence/stage2/s26](evidence/stage2/s26/03_RESULTS.md).

Review of that trial found one reporting defect: the curve figure selected rows by origin label
alone, so it merged the train and validation paths (36 rows, 18 duplicate `(model, alpha)`
pairs). It is fixed, with a regression fixture; the figures were regenerated from copies of the
saved artifacts without refitting or rescoring, and the trial's originals are kept as clearly
labelled superseded evidence. See [04_REPAIR.md](evidence/stage2/s26/04_REPAIR.md).

**Where the evidence stops.** The trial is execution and reuse evidence from one synthetic
series and one seed — not performance evidence, and not independent external validation.

The three PNGs were produced by an agent whose runtime returns image metadata only and exposes
no vision tool, so that agent **could not inspect them**; it checked that each file opens and
passes PIL `verify()` at its recorded dimensions, and that the renderer received the documented
points. A **separate reviewing assistant** afterwards inspected all three corrected figures on
2026-11-11 and recorded what it saw. That is assistant visual review, not human inspection, and
it does not replace the numerical checks — see
[03_RESULTS.md §5](evidence/stage2/s26/03_RESULTS.md) and
[04_REPAIR.md §7](evidence/stage2/s26/04_REPAIR.md#7-subsequent-visual-review--2026-11-11).

Skill loading is evidenced by the runtime's own skill notice; the trial's authorisation is
recorded as narrative, because no session transcript was archived.

## What this does not establish

- No leakage certificate. The verification checks are the ones listed in
  [docs/METHODS.md](docs/METHODS.md); trusted local plugin code is not sandboxed, and no check
  detects contaminated pretraining or researcher selection.
- No ranking, significance or suitability claim: comparisons are whole-method, differences
  carry no confidence interval, and repeated seeds on one series are not independent evidence.
- No ES-specific backtest and no bootstrap inference; both are unavailable, not passed.
- No regulatory certification, capital-adequacy, model-approval or suitability claim. This is a
  research prototype, not a risk-management control.
- Synthetic runs are implementation and execution checks, not evidence about markets.

## Licence, attribution and data

This project is licensed under the **[MIT License](LICENSE)**. [docs/LICENSING.md](docs/LICENSING.md)
records the licence rationale, the declared licences of the direct dependencies, and the
separation between code rights and market-data rights.

| Subject | Status |
|---|---|
| Project code | MIT License — see [LICENSE](LICENSE) |
| Required dependencies | `numpy`, `scipy`, `pandas`; each under its own licence, not vendored here |
| Optional dependencies | `arch` (`baselines`), `jax`/`optax` (`neural`), `matplotlib` (`plots`), `pandas_market_calendars` (`data`), `pytest` (`dev`) |
| Methods attribution | The dependency licence table and the method references are in [docs/METHODS.md](docs/METHODS.md) |
| Data | **Not redistributed.** `data/` is ignored; example CSVs are generated deterministically by code in this repository; the synthetic series come from the toolkit's own generator |
| Market-data rights | Separate from code rights. A user who ingests a vendor feed is responsible for that feed's terms; nothing here grants rights to it |

No third-party code or notices are vendored: the only model shipped without a toolkit import is
`examples/bring_your_model/ewma_gaussian.py`, which is project-written and documented as such.

Substantial AI assistance was used for planning and implementation. Reviewed code and saved
execution evidence, not the assistance, determine what is established.

## What the repository contains

`src/`, `tests/`, `docs/`, `configs/examples/`, `examples/`, `skills/`, `demo/` and
`evidence/stage2/s26/` are the release, together with `LICENSE` and `requirements-tested.txt`. `outputs/` and
`data/` are gitignored: run directories and market snapshots are never committed, and
[outputs/README.md](outputs/README.md) explains the local-only policy. Two root documents — the
2026-11-04 experiment protocol and the FZ0 decomposition note — are internal development records
kept for provenance; the run directories and data snapshots they name are not distributed, so
their commands are not reproducible from a clone.

`demo/index.html` is a self-contained offline presentation built from the S2.6 run; it needs no
installation or network. [demo/README.md](demo/README.md) documents its controls and the honest
boundary of its regeneration: refreshing the embedded numbers needs a local copy of the documented
EWMA run, which is reproducible but not distributed.
