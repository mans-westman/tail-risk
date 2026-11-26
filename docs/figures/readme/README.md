# README figures

`workflow.svg` and `ewma-use-case.svg` are embedded in the root README. Their PNG siblings
are 2× raster previews (1280 px wide). `generate.py` is the editable source for both figures.
All assets have an explicit opaque background; SVG lettering is stored as paths, so no fonts,
scripts or external assets are needed to display them. The palette follows
[the presentation](../../../demo/index.html); DejaVu Sans replaces its system-font stack for
portable scientific plotting. Muted labels are slightly brighter for small-screen contrast.
The compact workflow uses plain-language descriptions and a schematic report thumbnail (not
additional result data). The example foregrounds the plot and explains EWMA and the orange
markers; its displayed 2.4% is computed as 12/499, rounded to one decimal percent. Nominal
2.5% comes from the saved alpha. Introductory caveats live in the root README’s limitations
section rather than in the artwork.

## Regenerate without training

From the repository root, with the existing `plots` extra installed:

```sh
python docs/figures/readme/generate.py
```

This reads the shipped presentation's `script#evidence` JSON. It does not execute the HTML,
fit models, evaluate losses, alter the saved run, or replace the corrected S2.6 evidence.
The default build works from a clone without `outputs/`. Matplotlib generates the diagram
and scientific plot; numeric coordinates are never manually approximated or smoothed.

An optional read-only cross-check against the original local run is:

```sh
python docs/figures/readme/generate.py --check-run outputs/s26_gaussian_ewma
```

That path is gitignored and not needed for the default build. A different run is deliberately
rejected if its data do not match the shipped demonstration.

## Data and provenance

- [demo/build_demo.py](../../../demo/build_demo.py) documents how the saved forecasts and
  metrics were embedded in [demo/index.html](../../../demo/index.html).
- The plot selects `example_ewma_gaussian` on `example_garch_gaussian_validation` at
  `alpha=0.025`, using the intersection of admissible origins for EWMA, GARCH-t and
  historical simulation: **499 origins, indexed 1000–1498**. Each plotted return is the
  next-session target for the labelled forecast origin.
- The demo stores returns, VaR and ES in percent. The figure divides these stored values by
  100 to restore **signed decimal daily log returns**. It plots all 499 points, without
  downsampling, interpolation or smoothing. Amber markers denote `return <= VaR`: **12**.
- The original local sources are `outputs/s26_gaussian_ewma/forecasts.csv` and `metrics.csv`.
  The public [results record](../../../evidence/stage2/s26/03_RESULTS.md) gives the population,
  counts and numerical results. The [demo guide](../../../demo/README.md) explains its data
  extraction. The [repair record](../../../evidence/stage2/s26/04_REPAIR.md) distinguishes
  corrected from superseded figures.
- This is a new matched-sample redraw, not a crop of the corrected audit return figure.
  That figure uses model-specific available rows: EWMA **43/1,437**. Those points are not
  mixed into this **12/499** plot.
- The S2.6 data are synthetic Gaussian GARCH, with configured seeds 6201/6202, train 3000
  and validation 1500; no test split. This is model integration/reuse evidence, not evidence
  of superiority, calibration or market performance. Existing report limitations still apply.

## Verification performed — 2026-11-25

**Programmatic:** the generator checked unique, ordered common origins; finite values and
signed VaR/ES ordering; identical realized returns across models; metric counts and all three
models' matched exceedance counts. The optional local cross-check passed for all **1,497**
model/origin rows (three models × 499) and the primary-alpha metrics, with exact equality in
the demo's percent representation. It separately confirmed EWMA's available count 43/1,437.
No fitting, rescoring or numerical source edits were performed.

**Visual:** assistant inspection of the rendered figures at 640 px README width and 390 px
narrow-screen width checked labels, arrows, legends, clipping and contrast. This is visual
inspection of local renders, not human review or a live GitHub rendering test. Fine plot
labels are smaller on narrow screens; the full-size SVG is available by opening the image.
