# Diagnostic 2026-11-04 — where the joint MLP loses at alpha = 0.005 and 0.010

Status: explanation of saved results. No model, budget, feature, loss or selection rule was
changed, no run was repeated, and no observation was removed from any reported figure.

Scope: the real-data validation split (1,006 forecast dates, 2016-01-04 … 2019-12-31), the
two smallest alpha levels, all three model seeds (101/102/103), against both baselines.
Test-period data is not touched.

Script: `scripts/diagnose_fz0.py`. Reproduction:

```sh
JAX_ENABLE_X64=1 PYTHONPATH=src python3 scripts/diagnose_fz0.py \
    --run outputs/exp-20261104-real-s101 \
    --run outputs/exp-20261104-real-s102 \
    --run outputs/exp-20261104-real-s103 \
    --split validation --alphas 0.005 0.01 \
    --out outputs/exp-20261104-diagnostic
```

Outputs: `row_scores.csv` (per row: FZ0 and its three terms), `date_contributions.csv`
(per date and comparison: both models' scores, levels, exceedance flags and the difference),
`gap_summary.csv`, `level_summary.csv`, `top_date_recurrence.csv`, `decomposition.md`.

---

## 1. Corrections to the assessment delivered with the experiment

Five statements were too strong or wrong. Each is corrected here; the underlying numbers are
unchanged.

**1.1 "Well calibrated" — withdrawn.** The accurate statement is: *the reported tests did not
reject calibration at alpha = 0.025 on this validation sample* — 25 exceedances against 25.15
expected, Kupiec p = 0.976, Christoffersen independence p = 0.150 (seed 101; 27 and 29
exceedances for seeds 102 and 103, Kupiec p 0.712 and 0.448). That is a non-rejection on 1,006
dependent observations, not evidence of calibration, and it is weaker still because the
checkpoint was selected on this same validation metric. No claim of calibration is made
anywhere in this project.

**1.2 alpha = 0.005 has almost no resolving power in this sample.** At 1,006 validation
observations, alpha = 0.005 implies **5.03 expected exceedances** and alpha = 0.010 implies
10.06. A single day moves the exceedance count by 20% of the expectation at the smaller level,
and a single day's hinge penalty can exceed the whole mean gap. Section 3 quantifies this. The
three model seeds vary initialization and batch order only; they do **not** estimate the
sampling uncertainty of a 5-exceedance statistic, and nothing here does.

**1.3 Oracle errors differ by about one order of magnitude, not two.** At alpha = 0.025 the
GARCH–t VaR MAE is 9.0e-5 (Gaussian case) against 2.06e-3 for historical simulation and
2.27e-3 for the MLP: a factor of roughly 23 and 25, i.e. one order of magnitude, not two.
Also, the Gaussian case fits a *Student-t* GARCH to a Gaussian DGP: it approximates the truth
with a correctly specified volatility recursion and a flexible innovation density, and it is
not the exact data-generating model. "Correctly specified GARCH family" overstates it.

**1.4 The protocol's description of what W removes was wrong** and is corrected in
`EXPERIMENT_PROTOCOL_20261104.md` section 2. `required_context = W` decides the panel's single
`is_scored` flag per origin, which governs (a) which origins any model may forecast at and (b)
which origins are training-eligible for the MLP. It does **not** shorten the GARCH–t fit
sample: the runner calls `model.fit(train.returns)` on the whole train split array (6,553
returns here, 1,000 of them unscored origins). The real cost of W = 1000 is 900 forecast
origins per split and 900 MLP training origins, not 900 GARCH-fitting observations.

**1.5 The earlier test-period exposure is preserved, not minimised.** "Not used in this
experiment" is the accurate statement. It is not an untouched holdout: `outputs/sp500-real-20261102`
scored 2020–2025 with historical simulation and GARCH–t and its coverage diagnostics were
computed and inspected, as recorded in STATUS.md and `outputs/README.md`. What is true and
narrower: no joint-MLP forecast exists for any test session, and this experiment computed
none. The `exposure` string frozen into the ingest configs and run manifests predates that
history and understates it; it is not the exposure record.

Everything else in the earlier assessment stands: the MLP beats historical simulation at
alpha = 0.025 in all three cases, trails GARCH–t in all three, and is last on the real
equal-weight aggregate because of alpha = 0.005 and 0.010.

---

## 2. Arithmetic control

The decomposition recomputes each row's FZ0 from the saved `realized_return`, `var`, `es` and
`alpha`:

```
L = -1{y <= v} * (v - y) / (alpha * e)  +  v / e  +  log(-e) - 1
    \_____________ shortfall ________/     \_ratio_/  \_log-es_/
```

Two checks run before anything is written, and the script exits non-zero if either fails:

- per `(run, model, alpha)` the recomputed mean reproduces `metrics.csv::mean_fz0` to better
  than 1e-10 — measured worst disagreement 8.9e-16 over all 18 cells;
- the vectorised formula agrees with the scalar reference `tailrisk.evaluation.fz0_loss` to
  8.88e-16 on a 400-row sample per run.

Recognized after the fact: both checks compare numbers produced from the same saved table, so
they establish that this decomposition is *of the reported score*, not that the reported score
is right — the verifier's independent replay (0 mismatches over 39,342 MLP rows, 23/23 checks)
is what establishes that.

---

## 3. What the gap is made of

### 3.1 Forecast level and exceedances by alpha (all 1,006 dates, no mask)

Mean signed VaR/ES, decimal log returns. Conventional models are seed-invariant.

| model | alpha | mean VaR | mean ES | exceedances | expected | Kupiec p |
|---|---|---|---|---|---|---|
| historical | 0.005 | −0.02922 | −0.03498 | 4 | 5.03 | 0.633 |
| historical | 0.010 | −0.02436 | −0.03065 | 15 | 10.06 | 0.145 |
| historical | 0.200 | −0.00425 | −0.01103 | 179 | 201.20 | 0.076 |
| GARCH-t | 0.005 | −0.02350 | −0.02899 | 9 | 5.03 | 0.110 |
| GARCH-t | 0.010 | −0.02015 | −0.02530 | 17 | 10.06 | 0.045 |
| GARCH-t | 0.200 | −0.00604 | −0.01091 | 140 | 201.20 | 4.7e-7 |
| MLP (101 / 102 / 103) | 0.005 | −0.02215 / −0.02226 / −0.02321 | −0.02785 / −0.02731 / −0.02809 | 14 / 16 / 18 | 5.03 | 0.0010 / 9.6e-5 / 7.2e-6 |
| MLP | 0.010 | −0.02005 / −0.01985 / −0.02077 | −0.02529 / −0.02458 / −0.02540 | 17 / 20 / 20 | 10.06 | 0.045 / 0.0055 / 0.0055 |
| MLP | 0.200 | −0.00883 / −0.00870 / −0.00819 | −0.01149 / −0.01158 / −0.01120 | 97 / 96 / 109 | 201.20 | 4.1e-19 / 1.7e-19 / 5.8e-15 |

Two features stand out, and they point in opposite directions:

- **In level, the MLP's alpha = 0.005 VaR is not obviously too shallow.** Its mean
  (−0.0225) sits between GARCH–t (−0.0235) and historical simulation (−0.0292), yet it
  produces 14–18 exceedances against 4 and 9. The problem is *where* the shallow values
  fall, not the average depth.
- **The alpha curve is compressed.** Mean |VaR| from alpha = 0.005 to 0.200 falls by a factor
  of 2.63 for the MLP, 3.89 for GARCH–t and 6.88 for historical simulation; from 0.025 to
  0.005 the MLP deepens by only 1.35x against 1.48x and 1.58x. The network separates the four
  larger levels well and compresses the extreme one.

### 3.2 Exceedance structure at alpha = 0.005

Pooled over the three seeds (3 × 1,006 date-observations):

| comparison | MLP exceedances | baseline exceedances | both | MLP only | baseline only |
|---|---|---|---|---|---|
| vs historical | 48 | 12 | 12 | 36 | 0 |
| vs GARCH-t | 48 | 27 | 27 | 21 | 0 |

Every historical-simulation exceedance is also an MLP exceedance; the MLP adds 36. Of its 48
exceedances, 7 are days whose realised return is smaller than 1.5% in absolute value
(2017-03-02 −0.59%, 2017-06-20 −0.67%, 2018-01-29 −0.68%, 2018-01-30 −1.06%, …) where the
MLP's own VaR had fallen to −0.47% to −1.0%. The opposite failure appears on stressed days:
on 2016-06-24 the realised return was −3.66%, historical simulation said −2.53%, GARCH–t said
−1.96%, and the MLP said −1.16%.

So the same defect produces both: at alpha = 0.005 the conditional curve is too tight when
volatility is low and still too shallow when it is high. Both are failures of tail *shape*
across the conditioning, not a uniform level shift, which is why a single mean VaR comparison
looks acceptable.

### 3.3 Contribution of individual dates (per seed, per baseline)

`diff = FZ0(MLP) − FZ0(baseline)` on the same date; positive means the MLP scored worse.

| seed | alpha | vs | mean diff | worse / better dates | top-10 sum | top-10 share of positive diffs | mean diff with top-10 set to 0 |
|---|---|---|---|---|---|---|---|
| 101 | 0.005 | historical | +0.8057 | 204 / 802 | +1155.4 | 94.4% | −0.3429 |
| 101 | 0.005 | GARCH-t | +0.4989 | 370 / 636 | +627.5 | 93.1% | −0.1249 |
| 101 | 0.010 | historical | +0.2912 | 249 / 757 | +640.3 | 87.7% | −0.3453 |
| 101 | 0.010 | GARCH-t | +0.2078 | 471 / 535 | +301.2 | 80.1% | −0.0917 |
| 102 | 0.005 | historical | +0.8444 | 218 / 788 | +1181.3 | 91.8% | −0.3299 |
| 102 | 0.005 | GARCH-t | +0.5376 | 367 / 639 | +656.9 | 89.6% | −0.1154 |
| 102 | 0.010 | historical | +0.3301 | 255 / 751 | +679.8 | 85.5% | −0.3456 |
| 102 | 0.010 | GARCH-t | +0.2467 | 441 / 565 | +331.6 | 78.1% | −0.0829 |
| 103 | 0.005 | historical | +0.7647 | 256 / 750 | +1069.9 | 90.5% | −0.2989 |
| 103 | 0.005 | GARCH-t | +0.4579 | 458 / 548 | +559.8 | 84.7% | −0.0986 |
| 103 | 0.010 | historical | +0.3086 | 286 / 720 | +618.0 | 82.2% | −0.3057 |
| 103 | 0.010 | GARCH-t | +0.2251 | 513 / 493 | +284.0 | 69.6% | −0.0571 |

Read carefully: the MLP is **better than historical simulation on 75–80% of validation dates**
at alpha = 0.005, and its total difference on the non-exceedance dates is negative
(−331 to −367). The positive mean comes entirely from 12–18 dates carrying very large hinge
penalties. The "mean diff with top-10 set to 0" column is a concentration statistic, not a
score to report: it says the sign of the headline rests on ten dates, which is the strongest
possible argument that a 1,006-day sample cannot settle the alpha = 0.005 question.

### 3.4 Which FZ0 term carries it

Aggregate difference per term (sums to the mean difference; means over 1,006 dates):

| seed | alpha | vs | shortfall | ratio | log(−ES) | total |
|---|---|---|---|---|---|---|
| 101 | 0.005 | historical | **+1.1734** | −0.0396 | −0.3282 | +0.8057 |
| 101 | 0.005 | GARCH-t | **+0.5918** | −0.0177 | −0.0752 | +0.4989 |
| 101 | 0.010 | historical | **+0.5937** | −0.0057 | −0.2968 | +0.2912 |
| 102 | 0.005 | historical | **+1.2187** | −0.0183 | −0.3560 | +0.8444 |
| 102 | 0.005 | GARCH-t | **+0.6371** | +0.0035 | −0.1030 | +0.5376 |
| 103 | 0.005 | historical | **+1.1078** | −0.0082 | −0.3350 | +0.7647 |
| 103 | 0.005 | GARCH-t | **+0.5262** | +0.0137 | −0.0820 | +0.4579 |

The shortfall term carries the whole deficit and more; the log(−ES) term *favours* the MLP
because its ES is narrower (|ES| ≈ 0.0279 against 0.0350 for historical simulation), and the
ratio term is negligible. The MLP is penalised for missing VaR exceedances, not for a
mis-scaled ES, and its narrower ES partially offsets the penalty — which is a property of
this loss, not a defence of the forecast.

### 3.5 Dates that recur across every seed and both baselines

Ranked by how many of the 12 (seed, alpha, baseline) comparisons place the date in their
top-10 largest differences:

| date | lists | realised | MLP VaR (mean) | baseline VaR (mean) | MLP ES | baseline ES | mean diff | MLP exceeds |
|---|---|---|---|---|---|---|---|---|
| 2016-09-09 | 12 / 12 | −2.48% | −0.97% | −1.89% | −1.13% | −2.36% | +143.7 | 12 / 12 |
| 2016-06-24 | 12 / 12 | −3.66% | −1.23% | −2.12% | −1.58% | −2.58% | +135.7 | 12 / 12 |
| 2017-08-10 | 12 / 12 | −1.46% | −0.91% | −1.78% | −1.10% | −2.26% | +58.7 | 12 / 12 |
| 2017-05-17 | 12 / 12 | −1.83% | −1.06% | −1.86% | −1.35% | −2.33% | +57.1 | 12 / 12 |
| 2018-02-02 | 12 / 12 | −2.14% | −1.37% | −2.06% | −1.80% | −2.62% | +49.1 | 12 / 12 |
| 2019-05-07 | 12 / 12 | −1.66% | −1.29% | −2.35% | −1.59% | −2.84% | +31.5 | 12 / 12 |
| 2018-10-10 | 11 / 12 | −3.34% | −1.25% | −2.18% | −1.57% | −2.68% | +110.6 | 11 / 11 |
| 2018-02-05 | 7 / 12 | −4.18% | −1.99% | −2.44% | −2.41% | −3.08% | +60.2 | 7 / 7 |
| 2018-10-24 | 7 / 12 | −3.14% | −2.48% | −3.02% | −2.98% | −3.68% | +26.3 | 7 / 7 |
| 2019-08-05 | 6 / 12 | −3.02% | −2.33% | −3.05% | −2.78% | −3.63% | +32.4 | 6 / 6 |
| 2017-08-17 | 5 / 12 | −1.56% | −1.26% | −1.79% | −1.49% | −2.28% | +19.0 | 5 / 5 |
| 2017-03-02 | 4 / 12 | −0.59% | −0.44% | −2.03% | −0.66% | −2.53% | +30.0 | 4 / 4 |
| 2018-02-08 | 4 / 12 | −3.83% | −3.14% | −3.87% | −3.60% | −4.80% | +29.4 | 4 / 4 |
| 2019-03-22 | 2 / 12 | −1.92% | −1.65% | −2.29% | −2.03% | −2.87% | +10.9 | 2 / 2 |

Six dates appear in all twelve comparisons, and they are the well-known stress sessions of
the period (2016-06-24 Brexit referendum, 2016-09-09, the February 2018 and October 2018
sell-offs, 2019-05-07). One entry is different in kind: on 2017-03-02 the realised return was
−0.59%, historical simulation forecast −2.03% and the MLP forecast −0.44%, so an ordinary day
became an exceedance for the MLP alone. That is the "too tight when calm" failure in its
purest form and it appears in fewer comparisons precisely because the penalty is small.

---

## 4. What the decomposition shows, and what it does not

Shows:

1. The real-data alpha = 0.005 deficit is a **tail-shape** failure, not a level failure: the
   average depth is close to GARCH–t, but the curve is compressed (2.63x across the grid
   against 3.89x and 6.88x) and misallocated across the conditioning — too tight on calm days,
   too shallow on stressed ones.
2. The **shortfall term** carries essentially all of it; the ES level and the VaR/ES ratio do
   not. Any redesign aimed at the ES head would be aimed at the wrong term.
3. The magnitude of the FZ0 gap is **dominated by about ten dates** (69.6–94.4% of all positive
   differences), six of which are common to every seed and every comparison. At alpha = 0.005
   this is unavoidable arithmetic: the hinge penalty scales as 1/alpha, so one −3.7% day
   against a −1.2% VaR contributes ~254 points to a mean of ~0.8.
4. The finding is **stable across seeds** in structure (same dates, same term, same sign) even
   though the exceedance count and the mean gap move with the seed. Seed variability is not
   what separates the MLP from the baselines here.
5. The alpha = 0.010 deficit has the same origin as alpha = 0.005, with weaker amplification
   and less concentration (top-10 share 69.6–87.7%).

Does not show:

- That the MLP is worse overall on real data. It is better on 75–80% of validation dates at
  alpha = 0.005, and better at alpha = 0.025–0.200. It is worse on the mean of all six levels
  because the levels are averaged equally.
- That a fix would help. Nothing here was tested as an intervention; a shallower or deeper
  tail head, a per-alpha calibration, more capacity or more epochs are all untested, and the
  first of them is exactly the kind of change the one-experiment budget reserves for Tier 3.
- Anything about the test period, about a second index, or about the synthetic cases beyond
  what the earlier summary already reported.
- The statistical significance of a ~0.8 FZ0 gap on 1,006 dependent observations with 5.03
  expected exceedances. No such test was attempted and none should be read into these tables.

---

## 5. Consequence for Stage 1

The baseline-implementation gate does not require the MLP to win. Three baselines now run
through the same protocol, the same mask, the same evaluator and the same verifier, and the
comparison is reproducible end to end at ~7 minutes. Stage 1 is recorded as complete on that
basis (STATUS.md), with the alpha = 0.005 result reported as a property of the third baseline
rather than a blocking defect — and with the diagnostic above available as the starting point
if a later tier chooses to act on it.
