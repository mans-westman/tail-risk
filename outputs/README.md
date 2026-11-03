# Local run artifacts

Generated runs are ignored by Git. `run` refuses a nonempty output directory, so every run needs a fresh path; `verify`, `evaluate` and `report` intentionally overwrite only their own derived artifacts (`artifact_verification.json`, `metrics.csv`, `report.md`).

## Current evidence

- `hygiene-smoke-20261103/` — synthetic execution check under the current verifier: 7,848 forecasts, 32/32 offline checks, 19/19 artifact checks, replay 0 mismatches, 36 metric rows. Bit-identical to `repair-smoke-20261103` (maximum numeric delta 0.0 in both forecasts and metrics), so it documents that the 2026-11-03 hygiene pass moved no number.
- `sp500-dev-20261103-v2/` — **canonical development run** (`^GSPC`): `splits: ["train", "validation"]`, so the 2020–2025 test period is never forecast or scored. 89,484 forecasts (44,742 historical + 44,742 GARCH-t), 24 metric rows, panel SHA-256 `75301d1f144921de…`, 18/18 applicable checks passed with 1 not applicable (no oracle exists for market data). Use this for every model-development comparison.
- `sp500-dev-20261103/` — the same specification repeated minutes earlier, byte-identical configuration and panel hash. Kept as a run-guard artifact; cite `-v2` to avoid ambiguity.

## Exposure and verifiability of the real-data run

- `sp500-real-20261102/` — first real-data run (`^GSPC`, raw snapshot SHA-256 `8d599cd575e1602cbc82dcd6b31f4727aee0d389fc9ddb2cc186621c39580a1d`, retrieved 2026-11-02T10:28:55Z). Full-period protocol: cutoffs fixed in review (train ≤ 2015-12-31, validation 2016–2019, test 2020–2025), 100-session window, six alphas, two models, GARCH-t fitted once on train only. 8,965 scored origins → 107,580 forecast rows; 34/34 offline checks; 36 metric rows; `nan` oracle columns by design. **The test period has been scored and its Kupiec/Christoffersen metrics inspected** — it is an exposed retrospective, not an untouched holdout, and no superiority claim attaches to it.
- Its manifest carries no `panel_sha256` and its resolved config carries no `cutoffs`, because both guarantees postdate it. The current `verify` replays all 107,580 rows from the boundary the ingestion audit declares, reports 0 mismatches, and still exits 1 on exactly `panel_sha256_matches` and `effective_cutoffs_frozen`. Read that as *cannot prove which panel it consumed*, not *numbers wrong*; the run's saved `artifact_verification.json` (15/15 at creation) is the evidence of record. Do not patch the manifest to silence this.
- Replay is only comparable under the dependencies a run was produced with: this run (numpy 2.5.3, pandas 3.0.5) reproduces exactly there, while numpy 2.5.4/pandas 3.1.0 shifts the fitted GARCH parameters in the 6th–7th digit and reports 53,790/53,790 GARCH mismatches. `verify` names such drift in its replay detail so it is not misread as data corruption. Fresh real-data evaluation requires separate authorization and a new run directory.

## Superseded or historical

- `mask-smoke-20261102/` — same forecasts and mask as `repair-smoke`, but its Christoffersen columns were computed with the pre-repair pooled probability (`(n01+n11)/n`) and without chain-break handling. Its FZ0, quantile loss and Kupiec values were unaffected; cite a later synthetic run for any diagnostic.
- `consolidation-smoke-20261102/` — identical forecasts, but its `metrics.csv` predates the common scoring mask, so its per-model `n` and mean scores are not comparable across models.
- `repair-smoke-20261102/` and `repair-smoke-20261103/` — synthetic runs under the repaired evaluator; the 2026-11-03 one is the reference point for the neutrality comparison above. Both still verify.
- `smoke/`, `relocation-smoke-20261031/`, `verify-repair-20261031/` — single-model, pre-consolidation runs kept as historical evidence. `smoke/` also predates the 2026-10-31 forecast-timing repair, which is why its replay now fails; it must not be used for evaluation.

Synthetic runs are execution checks, not statistical evidence; the real run is descriptive evidence for one index, not a forecasting claim. Historical manifests and forecast files are never edited to make an old output look current — a corrected specification is a new run. Market data itself is never stored here: only derived artifacts, and `data/` stays ignored.
