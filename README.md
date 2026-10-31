# Tailrisk

A small Python workflow for next-session value-at-risk (VaR) and expected shortfall (ES) forecasts, with synthetic cases where the true conditional risk is known. Returns use signed decimal units and alpha denotes lower-tail probability.

**Maturity: early research prototype.** Implemented: Gaussian/Student-t GARCH simulation, analytical oracle curves, historical simulation, six causal features, a synthetic CLI and numerical tests. The CLI currently forecasts with historical simulation only. Fitted GARCH files are unintegrated drafts; neural models, real-data evaluation, and the reusable WorkSwarm skill are not complete. No predictive superiority is claimed.

## Repository map

```text
tailrisk/                       # run commands here; intended Git root
  README.md                     # purpose, setup, current capabilities
  AGENTS.md                     # agent working rules
  PROJECT_GUIDE.md              # scientific specification and staged plan
  STATUS.md                     # current evidence, limitations, next step
  DECISIONS.md                  # short decision history
  LITERATURE_REVIEW.md           # incomplete literature audit
  GIT_WORKFLOW.md                # milestone-based Git guidance
  pyproject.toml                # package metadata and dependencies
  configs/smoke.json             # tiny synthetic development configuration
  src/tailrisk/                 # package and CLI
    models/                    # historical adapter and unintegrated GARCH drafts
  tests/                       # numerical and timing checks
  outputs/                     # local generated runs; ignored except README
```

The surrounding course workspace is not part of this repository. Course assignment documents and agent memory/session databases stay outside it. This package does not require those documents to run. Add `examples/`, a reusable `skills/` folder, and report sources only when those artifacts exist.

## Setup

From this directory, use a Python environment compatible with `pyproject.toml`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The current test suite imports the draft GARCH adapter, so the development extra includes `arch`. JAX/Optax are not yet required by working source. Dependency ranges are not a locked environment; see `STATUS.md` for the environment actually checked. A clean installation should be verified before release.

## Run and check

```sh
python -m pytest -q
python -m tailrisk run --config configs/smoke.json --output outputs/my-first-smoke
```

Choose a new output directory for every run: the current CLI does not yet protect existing outputs from overwrite. The smoke run is for execution checks, not statistical conclusions. Its partition named `test` is synthetic smoke data, not the project's final evaluation set.

The run writes `forecasts.csv`, separate `oracle_curves.csv`, a resolved configuration, manifest, mathematical checks, and a summary. Oracle fields are evaluation-only. The legacy `verify` command currently reruns generic mathematical checks; it **does not validate the run artifacts** and must not be treated as certification of the saved forecasts. Repairing this is the next task.

## Development and reuse

Read `STATUS.md` before extending the code. Use one active writer, a small scoped change, relevant tests, and a brief decision entry when the scientific or execution contract changes. For a bug report, include the input/config, command, environment, expected/actual behavior and a minimal lawful example; omit secrets and market-data caches.

This is a course project intended to become a reusable [openJiuwen WorkSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) workflow. Actual fresh-run skill evidence remains to be collected. Substantial AI assistance has been used for planning and implementation; reviewed code and saved execution evidence determine what is established. References and reuse distinctions are in `PROJECT_GUIDE.md`.

No project license has been selected yet. Resolve licensing and third-party attribution before public release; model/data sources have their own terms. Neither these forecasts nor the current prototype establish suitability for trading or regulatory use.
