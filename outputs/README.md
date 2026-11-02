# Local run artifacts

Generated runs are ignored by Git. Use a new directory for each run; the current CLI can overwrite an existing directory.

## Current evidence

- `sp500-real-20261102/` — **first real-data run** (`^GSPC`, snapshot SHA-256 `8d599cd575e1602cbc82dcd6b31f4727aee0d389fc9ddb2cc186621c39580a1d`, retrieved 2026-11-02T10:28:55Z). Frozen protocol: cutoffs fixed in review (train <= 2015-12-31, validation 2016-2019, test 2020-2025), 100-session window, six alphas, two models, GARCH-t fitted once on train only. 8,965 scored origins → 107,580 forecast rows; 34/34 offline checks; 15/15 applicable artifact checks plus one not-applicable (no oracle exists for market data); deterministic replay from the panel with 0 mismatches. `metrics.csv` has 36 rows and `nan` oracle columns by design. Descriptive only — no superiority claim.
- `repair-smoke-20261102/` — synthetic development run under the repaired evaluator: 7,848 forecasts, 30/30 offline checks as emitted at the time, 17/17 artifact checks, replay with 0 mismatches, 36 metric rows. The protocol refactor left it bit-identical (re-running `configs/smoke.json` reproduces these forecasts and metrics exactly); runs made now carry two extra offline checks, so a fresh synthetic run reports 32/32.

## Superseded or historical

- `mask-smoke-20261102/` — same forecasts and mask as `repair-smoke`, but its Christoffersen columns were computed with the pre-repair pooled probability (`(n01+n11)/n`) and without chain-break handling. Its FZ0, quantile loss and Kupiec values are unchanged by the repair; cite `repair-smoke-20261102` for any diagnostic.
- `consolidation-smoke-20261102/` — identical forecasts, but its `metrics.csv` predates the common scoring mask, so its per-model `n` and mean scores are not comparable across models.
- `smoke/`, `relocation-smoke-20261031/`, `verify-repair-20261031/` — single-model, pre-consolidation runs kept as historical evidence; `smoke/SUPERSEDED.md` explains why it must not be used for evaluation.

Synthetic runs are execution checks, not statistical evidence; the real run is descriptive evidence for one index, not a forecasting claim. Do not edit historical manifests to make old outputs look current. Market data itself is never stored here — only derived artifacts, and `data/` stays ignored.
