# Tested environment

What this page is: the environment in which the commands in the top-level `README.md` and in
`examples/README.md` were actually run, with the runtimes measured there. It is a record, not
a lock file and not a support statement.

What it is not: a guarantee for other platforms, interpreters or dependency versions.
`pyproject.toml` declares ranges, not pins; every run records the interpreter and the
dependency versions it actually used in `manifest.json`, and `verify` names dependency drift
between the verifying interpreter and a run's manifest when a refit is involved.

## Record

| Item | Value |
|---|---|
| Recorded | 2026-11-10; suite counts re-measured 2026-11-11 |
| Host | macOS 15.6, arm64 (`Darwin-arm64`) |
| Interpreter | CPython 3.13.15, virtual environment created with `python3 -m venv .venv` |
| Package | `tailrisk` 0.1.0, installed editable (`pip install -e ...`) |
| Shell | zsh; commands run from the repository root |

## Installation used

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[baselines]'      # quickstart: historical simulation + GARCH-t
```

Extras, and what each one makes possible:

| Extra | Installs | Needed for | Effects when absent |
|---|---|---|---|
| — | numpy, scipy, pandas | everything: import, `--help`, historical simulation, custom adapters | — |
| `baselines` | `arch` | `garch_student_t` | configs listing it stop with an install hint; other models still run |
| `neural` | `jax`, `optax` | `joint_mlp` | `configs/examples/student_t_mlp.json` cannot run; the rest of the suite skips its MLP tests |
| `plots` | `matplotlib` | `report --plots` | `report` still writes Markdown/JSON; `--plots` fails with the install command |
| `data` | `pandas_market_calendars` | optional exchange-session cross-check in `ingest` | the check is recorded as skipped; ingestion does not guess a calendar |
| `dev` | `pytest`, `arch` | the test suite | — |

Installed versions in this record: numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, arch 8.0.0,
matplotlib 3.11.2, pytest 9.2.0. The MLP timings below used jax 0.12.0.1 and optax 0.2.8 with
`JAX_ENABLE_X64=1`, which the MLP path requires for float64 parity with the saved
checkpoints.

## Measured runtimes

Wall-clock time for one cold command on the host above, console script, no other load. These
are observations, not a promise: a slower machine or a first-run font cache can multiply the
figure.

| Command | Run | Seconds |
|---|---|---|
| `run --config configs/examples/student_t.json` | 41,232 forecast rows, 2 models | 8.6 |
| `verify --run <student_t run>` | 27/27 applicable checks | 6.5 |
| `evaluate --run <student_t run>` | 4 tables, 24 cells | 1.8 |
| `report --run <student_t run>` | `report.md`, `report.json` | 0.8 |
| `report --run <student_t run> --plots` | the same plus 3 PNGs | 5.0 |
| `run --config configs/examples/gaussian.json` | 41,232 forecast rows, 2 models | 9.1 |
| `run --config configs/examples/student_t_mlp.json` | 67,476 rows, 3 models | 21.2 |
| `verify --run <student_t_mlp run>` | 35/35 applicable checks, checkpoint replay | 10.9 |
| `evaluate` / `report` on the MLP run | 36 cells | 2.7 / 1.1 |
| custom-adapter example: `ingest` then `run` | 420-row CSV, 3 models, 5,742 rows | 1.3 then 1.9 |

The Student-t quickstart therefore takes roughly 18 seconds end to end without figures and
about 23 with them.

## Test suite

| Environment | Extras installed | Result |
|---|---|---|
| fresh venv (scratch directory, not committed) | `dev` + `plots` (`dev` already supplies `arch`); **no** `neural`, **no** `data` | **411 passed, 10 skipped in 16.7-24.5 s** |
| development environment | `dev` + `neural` + `plots`; **no** `data` | **537 passed, 1 skipped in 67.3-180.8 s** |

The fresh environment was built by installing `.[baselines]`, then `.[plots]`, then `.[dev]`, so
the net extras are `dev` + `plots`. Its 10 skips are explicit: six `joint_mlp` adapter-parity
cases, one MLP core module, two MLP training/integration modules, and the exchange-calendar
cross-check. None of them fails silently.

The development environment has JAX and Optax, so the MLP tests run there; its single skip is the
calendar cross-check, because `pandas_market_calendars` (the `data` extra) is not installed in it
either. No environment in this record has `data` installed, which is why that check is skipped
in both rather than passing.

## Reproducing the record

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots,neural]'   # matches the development row; add ,data to exercise the calendar check
python -m pytest -q
```

`plots` is not decoration here: the figure tests call `report --plots` and expect it to succeed,
so without Matplotlib they fail rather than skip. `neural` is what turns the 10 skips into one.
Installing `.[dev]` alone therefore does not reproduce either row above.

If the suite reports a different pass count, the difference is either a missing optional extra
(the skip lines name it) or a changed dependency version; compare `pip freeze` against the
versions above before treating it as a code change.
