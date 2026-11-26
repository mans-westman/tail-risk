# Three-slide presentation

## Viewing

Open `index.html` in a modern browser. It is a self-contained offline presentation: no
installation, server, external fonts, CDN, or network request is required. Copy only that HTML
file to the presentation computer. Use full screen and a landscape display; the 16:9 canvas
scales to fit without cropping.

The three slides:

1. **Tail regimes** — Normal, Student-t and extreme-t densities with the 2.5% VaR held at −2%
   while ES moves from −2.39% to −2.74% to −6.07%. A controlled illustration of tail shape, not
   a fixed-volatility comparison or a market forecast.
2. **Workflow** — the run → verify → evaluate → report circuit, the owning causal input
   boundary, joint FZ0 scoring and explicit verification coverage.
3. **EWMA integration audit** — the fresh-agent trial's matched scorecard and the three
   corrected audit figures, redrawn from the saved data.

### Controls

- Space / right arrow / Page Down: advance. Slide 2 advances through six stages before moving to slide 3.
- Left arrow / Page Up: go back through stages or slides.
- 1, 2, 3: jump to a slide. Click any circuit stage to focus it.
- R: cycle tail regimes on slide 1. P: cycle audit plots on slide 3.
- F: full screen. B: black screen. Escape: restore the screen or dismiss a dialog.
- N: speaker notes. Enter activates a focused control. `?` opens keyboard help.

## Where the numbers come from

`index.html` embeds validation data from the fresh-agent EWMA comparison run and the three
reviewed PNGs from `evidence/stage2/s26/figures/`; both are already inside the file, so
**viewing needs nothing else**. Numerical results and limitations are recorded in
`evidence/stage2/s26/03_RESULTS.md`; the curve-selection repair is recorded in `04_REPAIR.md`.

The presentation return plot shows EWMA on the 499 matched validation origins, matching the
scorecard. The original audit return figure uses each model's own available population. Its
EWMA count is therefore 43 / 1,437, compared with 12 / 499 in the presentation. Coverage uses
the matched validation metrics. Risk curves use only validation origin 1498.

## Rebuilding

The builder reads a **local run directory**, so regeneration is reproducible but not
self-contained: `outputs/` is gitignored and no run is distributed. To rebuild the inputs from
this checkout, run the documented EWMA example into the path the builder expects, then rebuild:

```sh
python -m tailrisk run      --config configs/examples/gaussian_ewma.json --output outputs/s26_gaussian_ewma
python -m tailrisk verify   --run outputs/s26_gaussian_ewma
python -m tailrisk evaluate --run outputs/s26_gaussian_ewma
python3 demo/build_demo.py
```

`run` refuses a non-empty directory, so use that path only when it does not already exist. The
builder uses only the Python standard library, integrates Normal/Student-t densities, solves the
tail quantile, checks analytic ES against an independent numerical tail integral, verifies
matched counts and curve uniqueness, and embeds the saved evidence. It never fits or scores
models. It asserts the 499 matched origins and six alpha levels per model, so a materially
different run stops the build instead of silently changing the slides.

The embedded original figures are the reviewed files in `evidence/stage2/s26/figures/`, not a
copy under `outputs/`. Regenerating a run does not replace them, and the builder never writes to
`evidence/`; updating those figures is a separate, reviewable act.
