# Superseded trial figures

These are the three PNGs written by `report --plots` during the S2.6 trial on
`outputs/s26_gaussian_ewma`, kept unchanged so the defect they contain stays inspectable.

They were replaced in `../` on 2026-11-11 after a post-trial review found that
`curves_last_common_origin.png` selected rows by origin label alone. Because the train and
validation paths share numeric origin labels, that figure drew 36 rows at origin 1498 — two
values per model per alpha — and presented the result as a single validation-split curve.

`realized_var_es_primary_alpha.png` did not plot anything wrong; its title and caption were
silent about which population the panels use (each model's own available rows, not the common
mask). It is kept here because it is the file the trial produced.
`coverage_rate_vs_alpha.png` is byte-identical to its replacement and is kept only for
completeness of the trio.

Full record, including hashes and the regeneration procedure: `../04_REPAIR.md` and
`../03_RESULTS.md` §5.
