# Bring your own model: a standalone EWMA–Gaussian forecast

`ewma_gaussian.py` is a small, self-contained model a researcher can hand to a fresh agent as
the starting point for a custom adapter. It imports only NumPy and SciPy: **no `tailrisk`
import, no adapter, no factory registration, no checkpoint, no scoring or evaluation code.** That
is deliberate — what is being demonstrated is a model, not an integration, and the integration
is the part the toolkit's public contract is supposed to make routine.

## Interface

```python
from ewma_gaussian import forecast

var, es, sigma = forecast(history, alphas, *, decay=0.94, warmup=63)
```

| Argument | Meaning |
|---|---|
| `history` | one-dimensional finite signed decimal returns, **ending at the forecast origin**. Copied, never mutated. The target return must not be in it; the function cannot check that because it never sees what the caller passes. |
| `alphas` | lower-tail probabilities, each in `[0.005, 0.20]`. Query order is preserved in the output. |
| `decay` | fixed EWMA decay in `(0, 1)`, default `0.94`. A setting, not an estimated parameter. |
| `warmup` | observations used to initialise the variance, default `63`. `history` must hold at least `warmup` observations; with exactly that many, the forecast is the initialised variance and no update has been applied. |

Returns `(var, es, sigma)`: two float arrays aligned with `alphas`, in return units and negative
at these levels, plus the scalar next-session volatility forecast `sqrt(h)`.

## Mathematics

Zero conditional mean, one session ahead.

1. Initialise `h` as the mean squared return over the first `warmup` observations.
2. For each later return `r`: `h ← decay · h + (1 − decay) · r²`.
3. The final `h` forecasts next-session variance; `sigma = sqrt(h)`.
4. With `z = Φ⁻¹(alpha)`: `var = sigma · z` and `es = −sigma · φ(z) / alpha`.

`history` must therefore hold at least `warmup` observations. At **exactly** `warmup` there are no
later returns, so step 2 has not run: the forecast is the initialised mean-square variance. Each
additional observation applies the recurrence once. This is the same 63-observation floor the
toolkit uses, so a model declared with `minimum_history = 63` can forecast from a 63-return
causal history.

There is no training step. Volatility updates with new returns; the decay is not estimated, no
likelihood is maximised and no hyperparameter search runs. With `decay = 0.94` and `warmup = 63`,
63 returns of `0.01` followed by one or two returns of `0.02` give

```
h (63 observations)  = 0.01²                        = 1.0000e-4   initialisation only
h (64 observations)  = 0.94·1.0000e-4 + 0.06·0.02²  = 1.1800e-4   one update
h (65 observations)  = 0.94·1.1800e-4 + 0.06·0.02²  = 1.3492e-4   two updates
sigma = sqrt(1.3492e-4)                             = 1.161551e-2
alpha = 0.025 → var = −0.02276598, es = −0.02715476
```

Those are the numbers the self-check asserts, and they are derived by hand in the test, not
captured from a previous run.

## Validation

Inputs are validated and the function raises `ValueError` rather than clipping, dropping or
repairing anything:

| Condition | Result |
|---|---|
| history not one-dimensional, empty, or containing a non-finite value | `ValueError`, naming the offending index |
| `len(history) < warmup` | `ValueError`, naming both counts (initialisation needs `warmup` observations and no update is required) |
| `warmup < 2` or not an integer | `ValueError` |
| `decay` not strictly inside `(0, 1)` | `ValueError` |
| any `alpha` outside `[0.005, 0.20]` | `ValueError`, listing the offending levels |
| final variance non-positive or non-finite (e.g. an all-zero history) | `ValueError` |

## Check it

```sh
python examples/bring_your_model/ewma_gaussian.py
```

The self-check confirms initialisation, two applications of the recurrence, the Gaussian VaR/ES
outputs at the resulting `sigma`, order preservation under a reversed query, `es <= var` at every
level, the length boundary (62 observations rejected, 63 accepted with the initialisation-only
forecast, 64 accepted with exactly one update), and that each invalid input above is rejected. It
prints the sampled values.

## Attribution and scope

Project-written demonstration; no code is reused from `tailrisk` or from another package beyond
the NumPy/SciPy primitives it calls. The `0.94` decay is the daily convention popularised by the
RiskMetrics technical documentation (J.P. Morgan/Reuters, 1996), not a value fitted here. The
Gaussian VaR/ES expressions are the standard closed forms for a zero-mean normal with scale
`sigma`; they describe the model's own assumption, so on fat-tailed data this model is
misspecified by construction — which is a result to report, not a defect to tune away.

This model updates volatility but **does not estimate parameters**. It is therefore not
comparable on equal terms with a GARCH-t whose parameters were fitted on a train split, and
nothing here is a performance claim.

## Integrating it (not done here)

To run it inside the toolkit, write an adapter as described in
[the adapter contract](../../docs/MODEL_ADAPTER.md): `fit` records the training history it is
given (or nothing, since this model has no parameters), `predict_one` calls `forecast` with the
causal `history` and the queried `alphas` and maps the result to one `Forecast` per level with
`scale = sigma`, and `summary` reports the fixed settings. The `minimum_history` to declare is
`warmup`, i.e. `63`: at exactly that many observations the model already produces a forecast, so a
larger floor would reject histories it can serve. Registering it needs only a `custom_models` entry
naming `module:function`; no evaluator source change is required, and `replay_mode` can be `"none"`
because there is no checkpoint to replay.

That integration, the run and any comparison are intentionally **not** included here. They are
the work the fresh agent must do from the public documentation.
