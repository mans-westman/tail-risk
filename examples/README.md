# Examples

Five runnable cases and one standalone researcher-supplied model. All of them are offline:
nothing here downloads data, and no market snapshot is bundled.

Each example is an **execution and reuse check**. The synthetic ones have a known
data-generating process, which makes their oracle columns an implementation check, and a few
hundred scored origins, which at `alpha = 0.005` means a handful of expected exceedances. Do
not read a comparison here as performance evidence.

Run everything from the repository root, with the package installed
(`python -m pip install -e '.[baselines]'`). Every command writes into a fresh output
directory; `run` refuses a nonempty one.

## 1. Student-t GARCH, the primary quickstart

```sh
python -m tailrisk run      --config configs/examples/student_t.json --output outputs/example-student-t
python -m tailrisk verify   --run outputs/example-student-t
python -m tailrisk evaluate --run outputs/example-student-t
python -m tailrisk report   --run outputs/example-student-t
```

Add `--plots` to the last command (and install `'.[plots]'`) for the three PNGs under
`outputs/example-student-t/plots/`.

What it does: simulates a GARCH(1,1) process with Student-t(5) innovations — the toolkit's
existing generator and parameters, burn-in 2000 — then fits nothing on the test split because
none is declared. 3000 train and 1500 validation returns, `W = 1000`, six alpha levels, two
conventional baselines. Expect 41,232 forecast rows and 27 applicable verification checks on
a machine like the one in [the environment record](../docs/ENVIRONMENT.md).

## 2. Gaussian GARCH, the same estimators on thinner tails

```sh
python -m tailrisk run      --config configs/examples/gaussian.json --output outputs/example-gaussian
python -m tailrisk verify   --run outputs/example-gaussian
python -m tailrisk evaluate --run outputs/example-gaussian
python -m tailrisk report   --run outputs/example-gaussian
```

Identical settings, seeds 6201/6202 instead of 6101/6102, Gaussian innovations. The
GARCH-t fit is then misspecified by construction — a legitimate result to look at, and the
reason the two examples exist side by side.

## 3. Optional: the joint MLP

```sh
python -m pip install -e '.[neural]'
export JAX_ENABLE_X64=1
python -m tailrisk run      --config configs/examples/student_t_mlp.json --output outputs/example-student-t-mlp
python -m tailrisk verify   --run outputs/example-student-t-mlp
python -m tailrisk evaluate --run outputs/example-student-t-mlp
python -m tailrisk report   --run outputs/example-student-t-mlp
```

Same data and splits, plus the alpha-conditioned MLP with the training settings recorded for
the 2026-11-04 development experiment and no tuning. This is the only example that needs JAX,
and it takes roughly three times as long as the quickstart. `verify` replays the saved
checkpoint (35 applicable checks) instead of refitting.

## 4. Custom adapter: a declared CSV and your own model

Four commands, from the repository root:

```sh
python examples/custom_model/make_demo_returns.py
python -m tailrisk ingest   --config examples/custom_model/ingest.json --out outputs/example_local/dataset
python -m tailrisk run      --config examples/custom_model/run.json --output outputs/example_local/run
python -m tailrisk verify   --run outputs/example_local/run
```

`make_demo_returns.py` writes `examples/custom_model/data/returns.csv`: a deterministic
closed-form series (no RNG, same bytes every run) so the example needs no market data. The
ingestion audit records its SHA-256. The run then scores the CSV with historical simulation,
GARCH-t and `example_empirical`, a rolling empirical VaR/ES adapter defined in
`examples/custom_model/model.py` and registered only by the configuration.

Two practical points:

- use `python -m tailrisk`, not the installed `tailrisk` script, for this example. The factory
  is `examples.custom_model.model:create_model`, which resolves against the working directory,
  and `python -m` puts the repository root on `sys.path`. From another directory the run stops
  with an explicit "module is not importable" message rather than importing something else.
- the CSV is a declared input, not a feed: dates must be ascending and unique, units are
  declared, gaps are refused rather than repaired. See
  [the data contract](../docs/DATA_CONTRACT.md).

To write your own adapter, read [the adapter contract](../docs/MODEL_ADAPTER.md); the worked
example is small enough to copy.

## 5. Researcher-supplied model: a standalone EWMA–Gaussian forecast

`bring_your_model/ewma_gaussian.py` is not a run. It is a small, self-contained model — NumPy
and SciPy only, no `tailrisk` import, no adapter, no factory, no checkpoint — exposing
`forecast(history, alphas, *, decay=0.94, warmup=63)`. It exists so that integrating a model
through the public contract can be done from the documentation alone:

```sh
python examples/bring_your_model/ewma_gaussian.py   # numerical self-check, prints its values
```

See [bring_your_model/README.md](bring_your_model/README.md) for the interface, the fixed
mathematics, the validation rules and what it deliberately leaves out.

## 6. The integration built from it

`ewma_gaussian_adapter/model.py` is that contract filled in for the supplied function and
nothing else: `predict_one` calls `bring_your_model.ewma_gaussian.forecast` with the causal
history the runner hands over and maps its three return values onto `Forecast` records. The
supplied file is imported, not copied or edited. `decay` and `warmup` are settings, so `fit`
estimates no parameter and the checkpoint is a small JSON file.

```sh
python -m tailrisk run      --config configs/examples/gaussian_ewma.json --output outputs/example-gaussian-ewma
python -m tailrisk verify   --run outputs/example-gaussian-ewma
python -m tailrisk evaluate --run outputs/example-gaussian-ewma
python -m tailrisk report   --run outputs/example-gaussian-ewma --plots
```

The config is the Gaussian example plus `example_ewma_gaussian`, so the two baselines and the
data are unchanged and their scores can be compared with case 2. As in case 4, use
`python -m tailrisk` from the repository root: the factory is
`examples.ewma_gaussian_adapter.model:create_model`. Expect 67,476 forecast rows, 33/33
applicable verification checks and an adapter replay of 26,244/26,244.

For a full step-by-step walkthrough — including WorkSwarm skill invocation, expected
verification counts, and reading the matched-versus-available scorecard — see
[docs/TUTORIAL_EWMA.md](../docs/TUTORIAL_EWMA.md). This is the case a fresh WorkSwarm agent
executed in the S2.6 trial, and its results, figures and limitations are recorded in
[the trial evidence](../evidence/stage2/s26/03_RESULTS.md).

## Where the rest of the documentation is

| Page | Contents |
|---|---|
| [README](../README.md) | purpose, install, the commands, what each artifact means |
| [TUTORIAL_EWMA](../docs/TUTORIAL_EWMA.md) | worked EWMA walkthrough: CLI/WorkSwarm run, verification, matched scorecard, adapter anatomy |
| [METHODS](../docs/METHODS.md) | formulas, estimators, attribution, limitations |
| [DATA_CONTRACT](../docs/DATA_CONTRACT.md) | the returns CSV and the ingestion config |
| [MODEL_ADAPTER](../docs/MODEL_ADAPTER.md) | the Python model interface and checkpoints |
| [REPORTING](../docs/REPORTING.md) | the derived tables, the report sections and the figures |
| [ENVIRONMENT](../docs/ENVIRONMENT.md) | tested environment, measured runtimes, extras |
