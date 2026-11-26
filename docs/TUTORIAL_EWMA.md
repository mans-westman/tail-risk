# Worked tutorial: integrating and auditing an EWMA tail-risk model

This walkthrough runs an existing custom VaR/ES model (`example_ewma_gaussian`) alongside two
built-in baselines (`historical_simulation` and `garch_student_t`) on synthetic daily returns,
runs the artifact checks and replays supported forecasts, and shows how to read the
matched-sample scorecard without misreading warm-up exclusions.

It uses the exact code and configuration shipped in `v0.1.0`: no model math is changed, no
market data is downloaded, and the optional `neural` (JAX) extra is not required.

## 1. Setup

Clone the repository and run every command from the **repository root** (required so `configs/`
resolves and `python -m tailrisk` can import `examples.ewma_gaussian_adapter.model`):

```sh
git clone --branch v0.1.0 --depth 1 https://github.com/mans-westman/tail-risk.git
cd tail-risk
```

Choose one environment setup:

- **To reproduce the exact scorecard numbers quoted below** (Python 3.13.15 with the exact
  versions tested for `v0.1.0`, see [requirements-tested.txt](../requirements-tested.txt)):

  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements-tested.txt
  python -m pip install -e . --no-deps
  ```

- **For a lighter range install** (without `jax`/`optax`; requires Python `>=3.10`):

  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -e '.[baselines,plots]'
  ```

  Range installs may resolve newer dependency versions (for example `arch 9.x` instead of the
  pinned `arch 8.0.0`). Structural counts and verification checks stay the same, while fitted
  `garch_student_t` numbers can differ slightly from the pinned table below.

**Offline boundary:** after dependencies are installed, the **numerical CLI** (`run`, `verify`,
`evaluate`, `report`) runs entirely offline with no market-data download and no API key. If you
drive the same workflow through **WorkSwarm** using `skills/tail-risk-audit/`, WorkSwarm itself
may still require network access and LLM model credentials depending on your agent setup.

## 2. Run the comparison

### Option A — Direct CLI (offline after install)

Execute the four commands into a **fresh** output directory (`run` refuses a nonempty path so
prior results cannot be overwritten accidentally):

```sh
python -m tailrisk run      --config configs/examples/gaussian_ewma.json --output outputs/tutorial-gaussian-ewma
python -m tailrisk verify   --run outputs/tutorial-gaussian-ewma
python -m tailrisk evaluate --run outputs/tutorial-gaussian-ewma
python -m tailrisk report   --run outputs/tutorial-gaussian-ewma --plots
```

Use `python -m tailrisk`, not the bare `tailrisk` script, so Python adds the repository root to
`sys.path` and resolves the local adapter module `examples.ewma_gaussian_adapter.model`. On the
tested Apple Silicon environment (`requirements-tested.txt`, Python 3.13.15), these four CLI
commands took ~45.5 s in total (`run` 26.5 s, `verify` 12.5 s, `evaluate` 3.5 s, `report`
3.0 s); package installation and agent-assisted runs take longer.

### Option B — Via the WorkSwarm skill (`tail-risk-audit`)

Install the regular skill into your WorkSwarm skills folder and start a new session from the
repository root:

```sh
cp -R skills/tail-risk-audit ~/.jiuwenswarm/agent/workspace/skills/
```

Then ask WorkSwarm to run and audit the example:

> Use `tail-risk-audit` with `.venv/bin/python` from the repository root to run, verify,
> evaluate, and report (`--plots`) `configs/examples/gaussian_ewma.json` into
> `outputs/tutorial-gaussian-ewma`. Summarize the verification coverage, the matched validation
> scorecard at `alpha = 0.025`, and any difference between matched and available sample counts.

The skill invokes the exact same four CLI commands above; the agent reads saved artifacts rather
than calculating VaR/ES in prose.

## 3. Confirm verification outputs

Check the printed summary from each step against these expected counts:

| Step | Expected result |
|---|---|
| `run` | `partitions: {'train': 3000, 'validation': 1500}`; **`67,476` forecast rows** (`historical_simulation`: `14,988`; `garch_student_t`: `26,244`; `example_ewma_gaussian`: `26,244`); `offline checks: 32/32 passed` |
| `verify` | **`33/33` applicable checks passed**, `0` not applicable, `0` unavailable (`partial=False`); `0` failed rows and `0` invalid outputs |
| Adapter replay (inside `verify`) | `checkpoint_replay:example_ewma_gaussian`: **`26,244` compared, `0` mismatches**, recomputed from the saved checkpoint and causal histories with `fit` never called |
| `evaluate` | **`36` rows** in each per-model table (`metrics.csv`, `score_decomposition.csv`, `curve_diagnostics.csv`: `3` models × `2` splits × `6` alphas), and **`36` rows** in `paired_scores.csv` (`3` model pairs × `2` splits × `6` alphas) |
| `report --plots` | `report.md`, `report.json`, and **3 PNGs** under `plots/`: `realized_var_es_primary_alpha.png`, `coverage_rate_vs_alpha.png`, `curves_last_common_origin.png` |

Details of every artifact column and check status are in [REPORTING.md](REPORTING.md) and
[METHODS.md](METHODS.md).

## 4. Read the scorecard: matched vs. available samples

The main value of evaluating models inside `tailrisk` rather than ad-hoc scripts is visible as
soon as you open `metrics.csv` and `plots/realized_var_es_primary_alpha.png` side by side.

### Why `n_dropped_by_mask` is 938 (and why it is not a failure)

`historical_simulation` requires a `1,001`-return warm-up (`historical_window = 1000`), whereas
`garch_student_t` and `example_ewma_gaussian` each declare `minimum_history = 63`. To compare
every model on the **same forecast origins**, `evaluate` scores all three on the intersection
(the **common mask**):

| Split | `n` (common mask, all models) | `n_available` (EWMA / GARCH-t) | `n_available` (historical) | `n_dropped_by_mask` (EWMA / GARCH-t) | `n_failed_rows` / `n_invalid_output` |
|---|---:|---:|---:|---:|---:|
| `train` | 1,999 | 2,937 | 1,999 | 938 | 0 / 0 |
| `validation` | 499 | 1,437 | 499 | 938 | 0 / 0 |

The 938 dropped origins per split are the warm-up gap imposed by `historical_simulation`, **not**
adapter failures or invalid forecasts (`n_failed_rows = 0`, `n_invalid_output = 0`).

### Validation scorecard (`alpha = 0.025`) — pinned `v0.1.0` results

Values below were reproduced on `v0.1.0` with [requirements-tested.txt](../requirements-tested.txt)
(lower `mean_fz0` is better; signed log returns, so `es <= var < 0`):

| Model | `n` (matched) | `mean_fz0` | Exceedances | Expected |
|---|---:|---:|---:|---:|
| `example_ewma_gaussian` | 499 | `-3.751181` | 12 | 12.48 |
| `garch_student_t` | 499 | `-3.754682` | 8 | 12.48 |
| `historical_simulation` | 499 | `-3.693883` | 9 | 12.48 |

From `paired_scores.csv` on `validation` at `alpha = 0.025` (`model − comparator`; negative
favors the first model, descriptive only with no confidence interval):

- `example_ewma_gaussian − garch_student_t`: `+0.003501`
- `example_ewma_gaussian − historical_simulation`: `-0.057298`

<details>
<summary>Optional detail: training split and VaR backtest p-values (<code>alpha = 0.025</code>)</summary>

| Split | Model | `n` | `mean_fz0` | Exceedances | Expected | Kupiec `p` | Christoffersen `p` |
|---|---|---:|---:|---:|---:|---:|---:|
| `train` | `example_ewma_gaussian` | 1,999 | `-3.769819` | 48 | 49.98 | 0.776 | 0.881 |
| `train` | `garch_student_t` | 1,999 | `-3.783353` | 37 | 49.98 | 0.052 | 0.716 |
| `train` | `historical_simulation` | 1,999 | `-3.769025` | 51 | 49.98 | 0.884 | 0.049 |
| `validation` | `example_ewma_gaussian` | 499 | `-3.751181` | 12 | 12.48 | 0.891 | 0.284 |
| `validation` | `garch_student_t` | 499 | `-3.754682` | 8 | 12.48 | 0.170 | 0.609 |
| `validation` | `historical_simulation` | 499 | `-3.693883` | 9 | 12.48 | 0.295 | 0.565 |

On `train` (`alpha = 0.025`), paired `mean_fz0` differences are `example_ewma_gaussian − garch_student_t = +0.013534` and `example_ewma_gaussian − historical_simulation = -0.000794`.

</details>

### Matched table vs. per-model plot population

If you open `plots/realized_var_es_primary_alpha.png`, its panel for `example_ewma_gaussian`
reports **43** validation exceedances, whereas `metrics.csv` reports **12**. Both numbers are
right because they summarize different populations:

- **`metrics.csv` (matched sample):** `12` exceedances on the `499` common validation origins
  shared with `historical_simulation` (comparable across models).
- **`realized_var_es_primary_alpha.png` (each model's own available rows):** `43` exceedances on
  all `1,437` validation origins that EWMA can serve (`34 / 1,437` for `garch_student_t`;
  `9 / 499` for `historical_simulation`).

The figure caption names its population explicitly so the two counts are never mixed.

## 5. Recorded trial evidence and optional baseline-invariance check

The four commands in Section 2 run only `configs/examples/gaussian_ewma.json`; they do **not**
run the two-model baseline config (`configs/examples/gaussian.json`).

In the recorded S2.6 fresh-agent trial ([03_RESULTS.md](../evidence/stage2/s26/03_RESULTS.md)),
a fresh WorkSwarm agent executed both configs and checked **baseline invariance**: all 24
baseline metric rows (`historical_simulation` and `garch_student_t`) matched between the two
runs under `pandas.DataFrame.equals` on the parsed `metrics.csv` tables (an equality check on
parsed rows, not a byte-level file comparison). If you want to reproduce that check locally,
run `configs/examples/gaussian.json` into a separate directory and compare the baseline rows:

```sh
python -m tailrisk run      --config configs/examples/gaussian.json --output outputs/tutorial-gaussian-base
python -m tailrisk evaluate --run outputs/tutorial-gaussian-base
```

The S2.6 record also documents a post-trial plotting bug discovered in review (train and
validation rows merged at origin 1498), its regression fix, and a subsequent assistant visual
review of the regenerated figures (assistant review, not human inspection): see
[04_REPAIR.md](../evidence/stage2/s26/04_REPAIR.md).

## 6. How the custom adapter works (for bringing your own model)

If you want to plug your own Python model into `tailrisk`, this example splits cleanly into
three files without editing the runner, evaluator, or verifier:

1. **The standalone model (`examples/bring_your_model/ewma_gaussian.py`):** imports only NumPy
   and SciPy — zero `tailrisk` imports — and exposes
   `forecast(history, alphas, *, decay=0.94, warmup=63) -> (var, es, sigma)`. You can verify its
   hand-derived math independently before any adapter runs:

   ```sh
   python examples/bring_your_model/ewma_gaussian.py
   ```

   See [examples/bring_your_model/README.md](../examples/bring_your_model/README.md) for the
   equations and input validation rules.

2. **The Contract v1 adapter (`examples/ewma_gaussian_adapter/model.py`):** imports and calls
   `forecast(...)` directly without reimplementing or modifying it:
   - declares a frozen `ModelSpec` (`model_id="example_ewma_gaussian"`, `minimum_history=63`,
     `uses_validation=False`, `replay_mode="checkpoint"`);
   - records partition metadata in `fit` (`decay=0.94` and `warmup=63` are fixed settings, not
     estimated parameters);
   - implements `predict_one(inputs, alphas)` using the read-only causal `inputs.history` copy
     ending at `r_t`, returning one `Forecast` per alpha in query order; and
   - writes and reloads a small JSON checkpoint (`save_checkpoint` / `load_checkpoint`) so
     `verify` can replay all `26,244` forecasts with `fit` disabled.

3. **Config registration (`configs/examples/gaussian_ewma.json`):** adds the model id to
   `models` and names its factory under `custom_models`:

   ```json
   "models": ["historical_simulation", "garch_student_t", "example_ewma_gaussian"],
   "custom_models": {
     "example_ewma_gaussian": {
       "factory": "examples.ewma_gaussian_adapter.model:create_model",
       "settings": {"decay": 0.94, "warmup": 63},
       "dependencies": ["numpy", "scipy"]
     }
   }
   ```

To adapt your own model or local CSV return series, see [MODEL_ADAPTER.md](MODEL_ADAPTER.md)
and [DATA_CONTRACT.md](DATA_CONTRACT.md).

## 7. What this tutorial does not establish

- **Not market evidence or predictive superiority:** the run uses two separate synthetic
  GARCH(1,1)-Gaussian paths (`train` seed `6201` and `validation` seed `6202`, each with its own
  2,000-step burn-in), with no test split declared. It is an integration and reuse check, not
  empirical market evidence.
- **Whole-method comparison, not isolated architecture:** comparing fixed-decay EWMA (`0.94`)
  against a GARCH(1,1)-Student-t fitted on `train` is a legitimate whole-method comparison, but
  it does not isolate model structure under matched estimation procedures.
- **Limited statistical inference:** paired `FZ0` differences carry no confidence intervals;
  Kupiec and Christoffersen are VaR-only diagnostics, and there is no ES-specific backtest or
  bootstrap inference.
- **Not a security sandbox or universal leakage certificate:** `PredictionInput` hands each
  adapter an owning causal prefix and `verify` runs the named checks in [METHODS.md](METHODS.md),
  but custom Python adapters remain trusted local code. This toolkit is a research prototype,
  not a regulatory certification.
