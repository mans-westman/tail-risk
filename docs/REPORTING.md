# Evidence report and derived tables

This page describes what `evaluate` and `report` write in S2.4, what each column means, and
what the output may and may not be used to claim. It matches `src/tailrisk/evaluation.py`
(derived tables) and `src/tailrisk/reporting.py` (rendering); where the code and this page
disagree, the code is right and this page is a bug.

The split of responsibilities is deliberate:

| Command | Does | Does not |
|---|---|---|
| `evaluate` | scores saved forecasts on one common mask, writes four tables | fit, refit, download, select a model |
| `report` | renders those tables into `report.md` and `report.json`, optionally three PNGs | score, fit, download, select, call an LLM |

Rendering is reproducible from the run directory alone. If a number in the Markdown cannot
be found in a saved table or check, it is a defect, not an approximation.

## The four tables

`metrics.csv` keeps its columns and its meaning. The other three are additive and are
derived from the *same* validated rows and the *same* common mask, so they cannot drift
apart from `metrics.csv`.

### `score_decomposition.csv`

Per `(series_id, split, model_id, alpha)`, using exactly the rows that `metrics.csv` scores
in that cell (`n` matches, and so does `mean_fz0`):

| Column | Definition |
|---|---|
| `mean_shortfall` | mean of `-max(v - y, 0) / (alpha * e)` |
| `mean_ratio` | mean of `v / e - 1` |
| `mean_log` | mean of `log(-e)` |
| `mean_fz0` | mean of the FZ0 loss, computed by `fz0_loss` |
| `decomposition_residual` | `mean_fz0 - (mean_shortfall + mean_ratio + mean_log)` |

The three terms are the additive parts of the FZ0 formula, not a new loss: the indicator
`1{y <= v}` is exactly "`v - y` is non-negative", and `v/e + log(-e) - 1` is
`(v/e - 1) + log(-e)`. Their means must sum to mean FZ0, because the decomposition is
linear. The residual is written to the table rather than silently discarded, and `evaluate`
fails when it exceeds `1e-12` at the scale of the largest term — a broken identity is a
coding defect, not a rounding artefact. No term is dropped, rescaled, or read as a cost of
capital.

### `paired_scores.csv`

Per `(series_id, split, model_id, comparator_id, alpha)`:

| Column | Meaning |
|---|---|
| `comparator_kind` | `conventional_baseline` for the non-neural built-ins |
| `n` | size of the common mask for that cell (the same origins for both models) |
| `mean_fz0_model`, `mean_fz0_comparator` | the two means on that mask |
| `mean_difference` | `mean_fz0_model - mean_fz0_comparator`; negative means the model is better because lower FZ0 is better |

Rows are emitted for every model against each conventional baseline present, with two
rules that keep the table readable:

- a model is never compared with itself;
- two conventional baselines are compared once, in configured roster order, rather than
  twice with opposite signs.

The second rule applies **only** when both sides are conventional baselines, because only
then are the two differences mirror images. A model that is not a baseline is never
suppressed: its difference to each baseline is the comparison the table exists to report, so
the roster order cannot remove it.

A cell whose mask is empty keeps its row with `n = 0` and empty means, so "nothing
comparable" is visible instead of absent. No standard error, confidence interval, p-value or
significance marker is produced, and `n` is not an independent sample size: repeated seeds,
overlapping windows and correlated series are not independent market observations.

### `curve_diagnostics.csv`

Per `(series_id, split, model_id, alpha)`. Two different questions, in different columns:

| Column | Meaning |
|---|---|
| `n_origins` | origins that emitted a row for this level |
| `n_admissible` | of those, rows with status `ok`, finite numbers and `es <= var < 0` |
| `n_inadmissible` | the rest |
| `n_grids_complete` | origins whose whole configured grid is present **and** admissible |
| `n_grids_incomplete` | origins with some levels missing or inadmissible; they enter no crossing count |
| `var_crossings_with_next` | of the complete grids, how many violate `var(alpha) <= var(next alpha)` |
| `es_crossings_with_next` | the same for ES |
| `n_pairs_with_next` | complete grids contributing to the crossing count (`0` for the highest alpha, whose crossings are empty) |

Within-alpha admissibility is a property of one row; a crossing is a property of a grid.
Comparing two of six levels is not a grid check, which is why incomplete grids are counted
and excluded rather than partially scored. The grid columns describe the model's whole
configured grid at that series and are therefore identical across its alpha rows.

A crossing-free grid means the reported levels are ordered. It is **not** a continuous
monotonicity proof, not an integrated-quantile identity, and says nothing about levels
outside the configured grid.

## The report

`report` writes `report.md` (primary) and `report.json` (schema version 1, the
agent-readable equivalent). The eight sections are fixed and ordered by
`docs/STAGE2_IMPLEMENTATION.md` §4:

1. **Experiment** — question, data and provenance, units, horizon, splits, exposure status,
   seeds, runtime, versions.
2. **Comparison conditions** — each model's estimator, selection rule, parameter count,
   minimum history, replay mode, information declarations; what is matched and what is not;
   whole-method versus controlled-architecture labelling.
3. **Integrity and verification coverage** — artifact checks, unavailable checks, offline
   checks, coverage counts, failure rows, mask exclusions.
4. **Performance by alpha** — `metrics.csv` rows, the FZ0 decomposition, paired differences.
5. **Exceedance diagnostics** — observed/expected counts, rate, Kupiec and Christoffersen
   outputs, transitions and chain breaks, with undefined cells left undefined.
6. **Risk-curve diagnostics** — the crossing and incompleteness summary.
7. **Synthetic accuracy** — oracle errors when an independent oracle exists, otherwise not
   applicable.
8. **Interpretation limits and artifacts** — the limits list, present/absent artifacts, the
   figures, and the exact `verify` / `evaluate` / `report` commands that reproduce the
   directory.

Each section ends with the artifacts it read. `report.json` carries the same content as
structured data; a JSON `null` means undefined, unavailable or not applicable — never zero.

Four things the report keeps apart on purpose: **integrity** (did the artifacts survive and
replay), **verification coverage** (which checks could run), **finite-grid structure** (do
the curves cross) and **statistical performance** (how the scores compared). No statistical
p-value contributes to an integrity verdict, and a clean integrity verdict is not evidence
of calibration.

Missing evidence stays unavailable. A legacy run evaluated before these tables existed still
renders; the derived sections then say *unavailable* and name `tailrisk evaluate` as the way
to produce them. An unreachable `data_audit.json` yields "units unavailable" rather than a
guess from magnitudes. The manifest records the configuration path only in
`config.resolved.json`, so the `run` command is not listed among the rerun commands; the
report says so.

## Figures

`report --plots` writes three fixed PNGs under `plots/`, using Matplotlib's `Agg` backend:

| File | Content | Omitted when |
|---|---|---|
| `realized_var_es_primary_alpha.png` | realized returns with VaR/ES and exceedance marks at alpha `0.025`, one panel per model | the run does not use `0.025`, or no admissible row exists at it |
| `coverage_rate_vs_alpha.png` | observed exceedance rate versus alpha with the nominal rate as reference, one panel per split | no cell has a nonempty scored sample |
| `curves_last_common_origin.png` | VaR and ES versus alpha at the last origin, **on one series**, whose grid is complete for **every configured model** | no such origin exists, or a configured model produced nothing admissible |

The primary alpha is fixed in advance. When a run does not use it the figure is omitted with
a stated reason rather than drawn at a friendlier level. Splits are never joined into one
line: their cells are different samples. The last-origin curves are a single descriptive
snapshot, not a representative or live forecast.

Origin labels belong to a series, not to a run: a train path and a validation path can both be
labelled `1498`. The curve figure therefore selects on **(series, split, origin)** and draws
exactly one value per model per configured alpha. Selecting on the label alone merged the two
paths and drew two values per alpha — a defect found in review after the S2.6 trial.

The two figures count different populations. The realized-return panels use each model's **own
available rows** at the primary alpha, and each panel title reports that panel's `n` and
exceedances; the scored tables, including the exceedance diagnostics, use the **common mask**.
A model with more available rows can therefore show more plotted exceedances than its
matched-sample count. That is a wider sample, not a different model.

Matplotlib is optional (`pip install 'tailrisk[plots]'`). Without it, `report` writes the
Markdown and JSON and `report --plots` fails with that installation command instead of
writing a report that silently lacks figures.

The roster for the curve figure is read from `config.resolved.json`, not from the rows that
survived: a model that was configured and produced nothing admissible makes the figure
unavailable with a reason naming it, rather than shrinking the plot to the models that
happened to work. Figures also apply the scorer's admissibility rule (`status == ok`,
finite, `es <= var < 0`), so an invalid output is never drawn.

## Reading the artifacts

Every reader — verification, evaluation, reporting — loads run CSVs through
`tailrisk.artifacts.read_artifact_csv`, which parses with pandas' round-trip precision.
Pandas' default parser can collapse two adjacent float64 values onto one, and that matters
exactly where the evidence is the number: a pair with ES one ulp above VaR violates
`es <= var < 0`, and a lossy read would silently score it as valid. The domain rule itself is
unchanged and existing artifacts are never rewritten.

## Not claimed

- No capital-cost, regulatory-suitability, head-causality or clipping-bias interpretation.
- No ES-specific calibration claim, and no calibration claim from a non-rejected sparse
  coverage test.
- No significance, ranking or model-selection claim from the paired or decomposed means.
- No claim that a verified checkpoint was fitted on permissible data; replay establishes
  reproducibility from saved state.
- No fixed total of checks anywhere: counts are read from the artifacts of that run.
