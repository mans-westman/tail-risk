# Tested environment

What this page is: the environment in which the commands in the top-level `README.md`, in
`examples/README.md` and in `skills/tail-risk-audit/` were actually run, with the runtimes
measured there. The exact package versions are pinned in `requirements-tested.txt` at the
repository root; this page records the host, what was measured, and where the boundaries of
the claim are.

What it is not: a guarantee for other platforms, interpreters or dependency versions.
`pyproject.toml` declares ranges, not pins; every run records the interpreter and the
dependency versions it actually used in `manifest.json`, and `verify` names dependency drift
between the verifying interpreter and a run's manifest when a refit is involved.

## Record

| Item | Value |
|---|---|
| Recorded | 2026-11-10; suite counts re-measured 2026-11-11 and again 2026-11-23 |
| Host | macOS 15.6, arm64 (`Darwin-arm64`) |
| Interpreter | CPython 3.13.15, virtual environments created with `python3 -m venv` |
| Package | `tailrisk` 0.1.0 — editable from the checkout for development, and as a built wheel for the clean-install check |
| Exact pins | `requirements-tested.txt` (repository root), 31 packages including the optional extras |
| Shell | zsh; commands run from the repository root unless stated |

## Installation used

Two routes, for two different readers:

```sh
# Library use: ranges from pyproject.toml, only what the quickstart needs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[baselines]'

# Reproducing the tested release: the exact pinned set, then the package with no re-resolution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
```

`--no-deps` is deliberate: the requirements file already contains every runtime, optional and
test dependency at the tested version, so a second resolution could move a pin.

### Tested, declared, and merely resolvable

These are three different claims and this release only makes the first one:

| Claim | Status |
|---|---|
| CPython 3.13.15 with the versions in `requirements-tested.txt` | **Tested** — the suite and all documented runs below were executed there |
| `requires-python = ">=3.10"` | **Declared** — the dependencies resolve on 3.10 (numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, arch 9.0.0, matplotlib 3.10.9, jax 0.6.2 as resolved on 2026-11-23); **no 3.10 interpreter was available here, so the suite was not run on it** |
| Any other platform, interpreter or dependency version | **Not claimed** — including Linux and Windows |

The CI workflow's `tests` matrix runs 3.10 and 3.13 on Linux. Until those jobs have actually run
remotely, 3.10 remains a declared range, not a tested one.

### Dependencies may resolve outside the pinned set

Because `pyproject.toml` uses ranges, a fresh `pip install -e '.[baselines]'` on 2026-11-23
installed `arch` 9.0.0, not the tested 8.0.0. That was checked specifically: with `arch` 9.0.0 a
41,232-row student-t run completed, its `verify` reported 27/27 applicable checks with 0
unavailable, and `tests/test_garch_model.py` plus `tests/test_adapter_parity.py` passed. The
pinned 8.0.0 remains the version the release was measured on. A future `arch` release is not
covered by either check.

## Optional extras

Extras, and what each one makes possible:

| Extra | Installs | Needed for | Effects when absent |
|---|---|---|---|
| — | numpy, scipy, pandas | everything: import, `--help`, historical simulation, custom adapters | — |
| `baselines` | `arch` | `garch_student_t` | configs listing it stop with an install hint; other models still run |
| `neural` | `jax`, `optax` | `joint_mlp` | `configs/examples/student_t_mlp.json` cannot run; the suite skips its MLP tests |
| `plots` | `matplotlib` | `report --plots` | `report` still writes Markdown/JSON; `--plots` fails with the install command |
| `data` | `pandas_market_calendars` | optional exchange-session cross-check in `ingest` | the check is recorded as skipped; ingestion does not guess a calendar |
| `dev` | `pytest`, `arch` | the test suite | — |

Absence was tested, not assumed, in a wheel-installed environment with only numpy, scipy and
pandas:

| Action | Result |
|---|---|
| `python -m tailrisk --help` | exit 0 |
| historical-simulation-only run → verify → evaluate → report | all exit 0; 14,988 forecasts; 32/32 offline checks; 27 applicable artifact checks; Markdown and JSON report written |
| `report --plots` | exit 2, `plotting requires Matplotlib … install it with pip install 'tailrisk[plots]'` |
| a config listing `garch_student_t` | exit 2, `run failed: the GARCH-t model requires the arch package … pip install 'tailrisk[baselines]'` (stops before fitting or creating outputs) |
| a config listing `joint_mlp` | exit 2, `run failed: the joint MLP requires the optional neural extra (jax, optax) … pip install 'tailrisk[neural]' and set JAX_ENABLE_X64=1` (stops before fitting or creating outputs) |

The `data` extra was never installed in a tested environment, which is why the calendar check is
*skipped* in both suite rows below rather than passing. `pandas_market_calendars` is therefore
not pinned in `requirements-tested.txt`.

## Custom adapter imports

A configured factory is imported by ordinary `importlib` from the interpreter's `sys.path`; the
toolkit does not add the repository to that path. Running `python -m tailrisk` from the
repository root puts the root on `sys.path`, which is why
`examples.ewma_gaussian_adapter.model:create_model` resolves there and nowhere else. This was
confirmed both ways with the wheel installed:

| Working directory | Result |
|---|---|
| the repository root | the EWMA example ran: 67,476 forecast rows, 33/33 applicable checks, checkpoint replay 26,244/26,244 |
| a directory outside the checkout | exit 2, `custom factory module 'examples.ewma_gaussian_adapter.model' is not importable … run from a directory where it resolves` |

The adapter contract is unchanged: a factory is still `module:function`, arbitrary Python in the
runner's process, and nothing here sandboxes it.

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
| no `neural`, no `data` | core + `baselines` + `plots` + `dev` (numpy, scipy, pandas, arch 8.0.0, matplotlib 3.11.2, pytest 9.2.0) | **423 passed, 10 skipped in 74.8 s** |
| pinned env, with `neural` | `requirements-tested.txt`, `JAX_ENABLE_X64=1` | **549 passed, 1 skipped in 147.1–186.0 s** (two runs) |

The 10 skips in the first row are explicit: six `joint_mlp` adapter-parity cases, one MLP core
module, two MLP training/integration modules, and the exchange-calendar cross-check. None of
them fails silently. The second row removes nine of those skips by installing JAX and Optax; its
single skip is the calendar check, because the `data` extra is not in the pinned set either.

Earlier records of this suite read 423/10 and 537–549/1; the counts moved as tests were added.
The numbers above are from the 2026-11-23 release check.

## Release-verification commands (2026-11-23)

Run on the host above, in fresh environments outside the checkout:

```sh
# full pinned environment
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
JAX_ENABLE_X64=1 python -m pytest -q

# clean wheel, non-editable, run from outside the checkout
python -m pip wheel . --no-deps -w /tmp/tr-dist
python -m venv /tmp/tr-core
/tmp/tr-core/bin/python -m pip install /tmp/tr-dist/tailrisk-0.1.0-py3-none-any.whl 'arch==8.0.0' 'matplotlib==3.11.2'
cd /tmp/tr-work && /tmp/tr-core/bin/python -m tailrisk run \
  --config /path/to/tail-risk/configs/examples/student_t.json --output quickstart
```

The clean-wheel quickstart produced 41,232 forecast rows, 27/27 applicable checks with 0
unavailable and 0 failed, 24 metric rows and three PNGs — the same counts as the documented
quickstart. If your checkout lives inside a cloud-synced folder (such as a synced desktop or
Documents directory) where virtual-environment reads stall, create the environment outside the
synced tree (for example `~/.venvs/tailrisk`) and either invoke its interpreter directly or
symlink `.venv` to it.

## If the suite reports different numbers

If the suite reports a different pass count, the difference is either a missing optional extra
(the skip lines name it) or a changed dependency version; compare `pip freeze` against
`requirements-tested.txt` before treating it as a code change.
