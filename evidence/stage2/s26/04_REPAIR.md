# 04 — Post-trial repair and evidence cleanup

Status: reviewer-discovered reporting defect, repaired after the trial. No model was changed, no
parameter tuned, no experiment rerun, and no diagnostic added. The run directories in `outputs/`
were not modified; the figures were regenerated from copies of their saved artifacts.

## The instruction

Quoted verbatim from the project owner (2026-11-11):

> Close the S2.6 trial with a reporting repair and evidence cleanup.
>
> 1. Fix reporting._figure_curves: select the chosen series/path, split and origin, not origin
>    alone. Require the configured roster and complete admissible grid. Add a regression fixture
>    where train and validation share numeric origin labels but have different forecasts; assert
>    the plotted values belong only to the selected path and each model contributes exactly one
>    value per requested alpha.
>
> 2. Preserve the original trial figures as superseded evidence. Generate corrected
>    reports/figures from copies of the saved artifacts, without fitting or rescoring. Visually
>    inspect all three. Record this as a post-trial reviewer-discovered repair.
>
> 3. Clarify the figure populations: the return-path panels show each model's available rows,
>    whereas comparison metrics use the common mask. Thus EWMA's 43 plotted exceedances and 12
>    matched-sample exceedances describe different samples.
>
> 4. Clean the public evidence: preserve an actual skill-tool/session excerpt or screenshot
>    supporting loading and the approval gate, alongside the agent-written narrative; sanitize
>    personal absolute paths; resolve references to logs currently excluded by .gitignore;
>    correct "no failures in the whole trial" to acknowledge the unsupported pytest --timeout
>    attempt already recorded later.
>
> Do not change models, tune, rerun experiments or expand diagnostics.

## 1. The defect

`reporting._figure_curves` located its rows with `frame[frame["origin_date"] == origin]`, but
`reporting._last_common_complete_origin` had picked the origin *within a split* while returning
only the label. Origin labels are per path: `example_garch_gaussian_train` and
`example_garch_gaussian_validation` are both indexed 0..1999 in the S2.6 runs.

Measured on the trial's own artifacts:

| run | rows at origin 1498, old selection | duplicate `(model, alpha)` pairs | series returned |
|---|---|---|---|
| `s26_gaussian_baseline` | 24 | 12 | train + validation |
| `s26_gaussian_ewma` | 36 | 18 | train + validation |

So every model drew two values per alpha, alternating between a train forecast and a validation
forecast, and the caption named only the validation split. The y-axis values in the superseded
figure (`figures/superseded/curves_last_common_origin.png`) are therefore not a curve of either
path. Scores, metrics and every table were unaffected: evaluation selects on `(series, alpha)`
and never used this path.

## 2. The repair

`src/tailrisk/reporting.py`:

- `_complete_origins` now requires each configured alpha **exactly once** at an origin
  (`nunique == size == len(alphas)`), so a duplicated grid is not "complete".
- `_last_common_complete_origin` is replaced by `_curve_selection`, which returns
  `(series_id, origin, split, rows)` and searches per series inside each split, in the existing
  split preference order. The chosen rows are exactly `series_id == series & origin_date ==
  origin`, so two paths cannot contribute to one curve.
- `_figure_curves` refuses to draw unless each configured model contributes exactly
  `len(alphas)` rows with `len(alphas)` distinct alphas; otherwise the figure is marked
  unavailable with a reason naming the model. The configured roster is still required from
  `config.resolved.json`, and a configured model with nothing admissible still makes the figure
  unavailable rather than being dropped.
- Title, axis note and caption now name the series and state "one value per model per
  configured alpha".

## 3. Regression fixture

`tests/test_reporting.py::test_curve_figure_selects_one_series_not_every_path_with_the_same_label`
builds a run where `ds_train` and `ds_validation` share origin labels 0..5 with deliberately
different VaR/ES, then asserts:

- the selection is `(ds_validation, origin, validation)`;
- every plotted row has `series_id == ds_validation`;
- `set((model, alpha))` equals the full configured cross-product and every count is 1;
- the plotted VaR/ES equal the validation rows (not the train rows) via `np.allclose`;
- `report --plots` still renders the figure and its caption names the series.

`tests/test_reporting.py`: 28 passed (was 27). The file's earlier roster and incomplete-grid
tests still pass unchanged. Full suite after the repair, in `tailrisk/.venv` (no `neural`
extra): **423 passed, 10 skipped in 18.25 s**. The 10 skips are the optional `jax`/`optax` and
`pandas_market_calendars` cases, unchanged by this work.

## 4. Regenerated figures

Each run was copied to `.tmp/s26_regen/{baseline,ewma}` and rebuilt with:

```
PYTHONPATH=src .venv/bin/python -m tailrisk report --run ../.tmp/s26_regen/ewma --plots
PYTHONPATH=src .venv/bin/python -m tailrisk report --run ../.tmp/s26_regen/baseline --plots
```

`PYTHONPATH=src` is deliberate: `.venv` holds a copied build (see `01_ENVIRONMENT.txt`), so the
plain `python -m tailrisk` used during the trial would have run the pre-repair code. Nothing was
fitted or scored; `report` only reads saved CSVs. Hash comparison of `forecasts.csv`,
`metrics.csv`, `score_decomposition.csv`, `paired_scores.csv` and `curve_diagnostics.csv` between
`outputs/s26_gaussian_ewma` and the regenerated copy: identical in all five.

Corrected captions now read (comparison run, `report.md` §8): "VaR and ES versus alpha at origin
1498 on series `example_garch_gaussian_validation` (validation split) … one value per model per
configured alpha".

Visual inspection was requested and **could not be performed**: this runtime returns metadata
only for image reads and configures no OCR or visual-question tool. Instead the renderer was
instrumented on the real artifacts and recorded six `plot` calls of six points each — one VaR
and one ES line per model, all x values in the configured grid — and all three PNGs pass PIL
`verify()` at the recorded dimensions. That is not a substitute for looking at them, and is not
reported as one.

## 5. Evidence cleanup

| Item | Change |
|---|---|
| Superseded figures | Trial PNGs moved to `figures/superseded/` with a README; corrected PNGs copied into `figures/`. Hashes and the reason for each are in `03_RESULTS.md` §5. |
| Personal absolute paths | Removed from `01_ENVIRONMENT.txt` and `02_COMMANDS.md`; the record is repository-relative, and `00_SKILL_LOADING.md` writes the home directory as `~` inside quoted tool output, marked as elided. |
| `.gitignore`d logs | `*.log` cannot be published, so `logs/` was removed from the evidence set and the decisive output lines are inlined in `02_COMMANDS.md`. The raw files are kept locally outside the repository's tracked tree; no reader-facing link points at them. |
| "No failures" | Corrected: the `pytest --timeout=900` attempt failed with `unrecognized arguments` (exit 4) before running any test, and the suite was rerun without the flag. Deliberately provoked adapter errors are listed separately. |
| Skill loading | `00_SKILL_LOADING.md` now carries the runtime's own availability notice verbatim plus a fresh inventory-scanner excerpt, instead of only a narrative claim. |
| Approval gate | The trial's authorisation wording is **not** reproduced: no transcript was archived, and inventing one would be worse than the gap. `00_SKILL_LOADING.md` states what is observable (the trial's boundary) and labels the rest as narrative. |

## 6. What this repair does not establish

- No pixel-level human confirmation of the three figures.
- No independent re-execution of the trial: the scores are the trial's, regenerated report
  included.
- No claim that the same label collision cannot appear elsewhere; this repair covers
  `_figure_curves`, which is where the review found it. `_figure_realized` already filters by
  split, and the S2.6 runs have one series per split, so it was not affected here.
