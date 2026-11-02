# Local run artifacts

Generated runs are ignored by Git. Use a new directory for each run; the current CLI can overwrite an existing directory.

Current evidence run: `repair-smoke-20261102/` — two models, common scoring mask, corrected coverage diagnostics, 30/30 offline checks, 17/17 artifact checks, deterministic replay with 0 mismatches over 7,848 forecasts, 36 metric rows. Its `metrics.csv` adds `n_transitions` and `n_chain_breaks`, and its Christoffersen values come from the repaired transition likelihood (D019, D020).

Superseded or historical:

- `mask-smoke-20261102/` — same forecasts and mask, but its Christoffersen columns were computed with the pre-repair pooled probability (`(n01+n11)/n`) and without chain-break handling. Its FZ0, quantile loss and Kupiec values are unchanged by the repair and still agree with the current run; cite `repair-smoke-20261102` for any diagnostic.
- `consolidation-smoke-20261102/` — identical forecasts, but its `metrics.csv` predates the common scoring mask, so its per-model `n` and mean scores are not comparable across models.
- `smoke/`, `relocation-smoke-20261031/`, `verify-repair-20261031/` — single-model, pre-consolidation runs kept as historical evidence; `smoke/SUPERSEDED.md` explains why it must not be used for evaluation.

All runs here are synthetic execution checks, not statistical evidence. Do not edit historical manifests to make old outputs look current. Curate small lawful examples separately when preparing a release.
