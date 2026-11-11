# 02 — Exact commands, exit codes and measured runtimes

Repository root: the checkout of `tail-risk` recorded in `01_ENVIRONMENT.txt`; every path below
is relative to it. Interpreter: `.venv/bin/python` (CPython 3.13.15).

Runtimes are wall-clock `real` values from `time -p`. The raw per-command logs were `*.log`
files, which `.gitignore` excludes from publication; they were therefore removed from this
record and the lines that carry the result are reproduced in "Output excerpts" below. Nothing
in this file depends on a file a reader cannot open.

One environment fact affects how these commands should be read: `.venv` holds a *copied* build
of `tailrisk`, not an editable link (`01_ENVIRONMENT.txt`), so `python -m tailrisk` imports
`site-packages/tailrisk`. The trial ran that copy. The reporting repair in `04_REPAIR.md` was
run with `PYTHONPATH=src` so that it exercised the working tree.

## Preflight

| # | Command | Exit | Time |
|---|---|---|---|
| 1 | `.venv/bin/python -m tailrisk --help` | 0 | <1 s |
| 2 | `.venv/bin/python skills/tail-risk-audit/check_references.py` | 0 | <1 s |
| 3 | `.venv/bin/python examples/bring_your_model/ewma_gaussian.py` (supplied self-check) | 0 | 0.89 s |

Self-check output (unchanged supplied file):

```
self-check passed: h0=1.000000e-04 h2=1.349200e-04 sigma=1.161551e-02
  lengths: 62 rejected; 63 -> h=1.000000e-04 (initialisation only); 64 -> h=1.180000e-04 (one update)
  alpha=0.2     var=-0.00977586  es=-0.01625950
  alpha=0.025   var=-0.02276598  es=-0.02715476
  alpha=0.005   var=-0.02991956  es=-0.03359145
  order preserved, es <= var on every level, invalid inputs rejected
```

## Adapter checks

| # | Command | Exit | Result |
|---|---|---|---|
| 4 | `.venv/bin/python -m pytest tests/test_ewma_gaussian_adapter.py -q` | 0 | 11 passed in 1.18 s |

## Run 1 — documented Gaussian example (unmodified config)

Config `configs/examples/gaussian.json`; output `outputs/s26_gaussian_baseline/`.

| Step | Command | Exit | Time |
|---|---|---|---|
| 5 | `python -m tailrisk run --config configs/examples/gaussian.json --output outputs/s26_gaussian_baseline` | 0 | 8.06 s |
| 6 | `python -m tailrisk verify --run outputs/s26_gaussian_baseline` | 0 | 6.85 s |
| 7 | `python -m tailrisk evaluate --run outputs/s26_gaussian_baseline` | 0 | 1.74 s |
| 8 | `python -m tailrisk report --run outputs/s26_gaussian_baseline --plots` | 0 | 2.01 s |

## Run 2 — comparison with the integrated adapter

Config `configs/examples/gaussian_ewma.json`; output `outputs/s26_gaussian_ewma/`.

| Step | Command | Exit | Time |
|---|---|---|---|
| 9 | `python -m tailrisk run --config configs/examples/gaussian_ewma.json --output outputs/s26_gaussian_ewma` | 0 | 10.94 s |
| 10 | `python -m tailrisk verify --run outputs/s26_gaussian_ewma` | 0 | 10.70 s |
| 11 | `python -m tailrisk evaluate --run outputs/s26_gaussian_ewma` | 0 | 2.52 s |
| 12 | `python -m tailrisk report --run outputs/s26_gaussian_ewma --plots` | 0 | 1.93 s |

Every numbered step above succeeded on its first execution and none of them was retried. That
is not the same as "no failures in the whole trial", which an earlier version of this file
claimed and which was wrong: one earlier attempt to run the regression suite used
`--timeout=900`, an option this pytest install does not provide. It failed with
`unrecognized arguments` and exit 4 before any test ran, and the suite was rerun without the
flag rather than worked around. Inside the adapter tests there were also deliberately
provoked errors (short history, zero variance, a malformed checkpoint), each asserted as an
expected outcome. The corrected statement is: no numerical command failed, and the
non-deliberate failures were an unsupported pytest flag and its rerun.

## Output excerpts

Copied from the removed `*.log` files, one line per command, with the run directory path as a
reader would see it:

```
run_baseline:   run complete -> outputs/s26_gaussian_baseline
                partitions: {'train': 3000, 'validation': 1500}
                forecasts: 41232 rows {'historical_simulation': 14988, 'garch_student_t': 26244}
                offline checks: 32/32 passed
run_ewma:       run complete -> outputs/s26_gaussian_ewma
                partitions: {'train': 3000, 'validation': 1500}
                forecasts: 67476 rows {'historical_simulation': 14988, 'garch_student_t': 26244,
                                       'example_ewma_gaussian': 26244}
                offline checks: 32/32 passed
verify_baseline: 27/27 applicable checks passed; 0 not applicable; 0 unavailable
verify_ewma:     33/33 applicable checks passed; 0 not applicable; 0 unavailable
evaluate_baseline: 24 metric rows; 24 decomposition rows; 12 paired rows; 24 curve rows
evaluate_ewma:     36 metric rows; 36 decomposition rows; 36 paired rows; 36 curve rows
report_baseline: report complete -> outputs/s26_gaussian_baseline/report.md and report.json
report_ewma:     report complete -> outputs/s26_gaussian_ewma/report.md and report.json
```

## Regression check (existing suite, unchanged by this trial)

| # | Command | Exit | Result |
|---|---|---|---|
| 13 | `.venv/bin/python -m pytest tests -q` | 0 | 422 passed, 10 skipped in 17.55 s |

The 10 skips are optional extras absent from this environment (`jax`/`optax` for the joint MLP,
`pandas_market_calendars` for the ingest calendar cross-check). The `--timeout=900` attempt
described above preceded this run; the count here is from the rerun without the flag.
