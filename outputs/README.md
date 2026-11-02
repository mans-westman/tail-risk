# Local run artifacts

Generated runs are ignored by Git. Use a new directory for each run; the current CLI can overwrite an existing directory.

Current evidence run: `mask-smoke-20261102/` — two models, common scoring mask, 30/30 offline checks, 17/17 artifact checks, 36 metric rows with coverage diagnostics.

Superseded or historical:

- `consolidation-smoke-20261102/` — identical forecasts, but its `metrics.csv` predates the common scoring mask, so its per-model `n` and mean scores are not comparable across models.
- `smoke/`, `relocation-smoke-20261031/`, `verify-repair-20261031/` — single-model, pre-consolidation runs kept as historical evidence; `smoke/SUPERSEDED.md` explains why it must not be used for evaluation.

All runs here are synthetic execution checks, not statistical evidence. Do not edit historical manifests to make old outputs look current. Curate small lawful examples separately when preparing a release.
