# Stage 2: reusable tail-risk research toolkit

**Approved direction:** 2026-11-05. **Status:** implementation specification, not a claim that these interfaces already work. README describes current capabilities. Complete the slices below in order; record evidence at each gate. This document fixes the Stage 2 architecture. Do not redesign it while implementing a slice; report a concrete incompatibility and proposed smallest amendment if a gate cannot be met.

## 1. Product and scope

For students and quantitative researchers, `tailrisk` compares conditional next-session VaR/ES models on a documented daily return series. It supplies common timing, evaluation, failure accounting, provenance, and inspectable reports. Humans and agents invoke the same Python implementation through the CLI. There are two journeys:

1. Run an offline example, supply a documented returns CSV, configure existing models and chronological splits, then run / verify / evaluate / report.
2. Implement a small Python model adapter, register it in configuration, and use exactly the same workflow without editing the evaluator.

Stage 2 supports one series per experiment, next-session decimal log returns, the existing Gaussian and Student-t GARCH generators, historical simulation, GARCH-t, the existing optional MLP, and one worked custom-adapter example. No model superiority is required. The constrained research model is a later optional adapter, not a dependency of delivery.

Excluded: live forecasting, trading or capital recommendations, dashboards, multi-asset pooling, arbitrary external feature feeds, periodic parameter refitting, hyperparameter search services, new DGP families, new research estimators, forecast-file-only imports, plugin marketplaces, and universal leakage certification. Do not add ES-specific backtests or bootstrap inference in this delivery stage. Report that those are unavailable.

## 2. Fixed architecture and ownership

Reuse numerical implementations; introduce only these focused seams:

| Location | Responsibility |
|---|---|
| `src/tailrisk/models/protocol.py` (new) | Public immutable input/output records and adapter protocol, independent of JAX/arch |
| `src/tailrisk/models/adapters.py` (new) | Wrappers for existing estimators; no duplicate loss, optimizer, oracle, or quantile implementation |
| `src/tailrisk/models/registry.py` (new) | Built-in factories plus explicit custom factory resolution |
| `src/tailrisk/runner.py` (new) | Permitted fitting data, per-origin prediction inputs, canonical forecast/failure rows |
| `src/tailrisk/data.py` | Existing ingestion/panel machinery plus local returns-CSV input |
| `src/tailrisk/evaluation.py` | Shared scoring mask and numerical statistics, including report tables |
| `src/tailrisk/verification_artifacts.py` | Generic artifact checks plus capability-specific replay; retain legacy handling |
| `src/tailrisk/reporting.py` (new) | Render saved results into Markdown/JSON and optional static plots |
| `src/tailrisk/cli.py` | Parse/validate arguments and call these modules; preserve existing command names |

This is not a general workflow framework. Extract only code needed by the seams. Preserve old APIs as compatibility wrappers until their callers/tests migrate. No mass renaming, estimator replacement, or unrelated cleanup.

### 2.1 Public adapter contract v1

Use dataclasses and a `typing.Protocol`; no extra validation dependency. All numerical arrays are NumPy arrays. Input arrays are owning, read-only copies, not slices whose `.base` exposes future data. Frozen dataclasses alone do not make arrays immutable.

```python
@dataclass(frozen=True)
class FitPartition:
    series_id: str
    path_id: str
    returns: np.ndarray             # one permitted path, including declared past context
    labels: tuple[str, ...]         # session dates or synthetic step labels
    target_indices: np.ndarray     # eligible observed targets; target j uses history [:j]

@dataclass(frozen=True)
class PredictionInput:
    series_id: str
    path_id: str
    origin_label: str
    history: np.ndarray            # ends at the origin, includes r_t, no r_(t+1)
    history_labels: tuple[str, ...]

@dataclass(frozen=True)
class Forecast:
    alpha: float
    var: float                     # signed decimal log-return units
    es: float                      # signed decimal log-return units
    scale: float | None = None     # optional causal diagnostic, never used to rescale again
    status: str = "ok"             # "ok" or "failed"
    reason: str | None = None

class TailRiskModel(Protocol):
    def fit(self, train: FitPartition,
            validation: FitPartition | None = None) -> None: ...
    def predict_one(self, inputs: PredictionInput,
                    alphas: np.ndarray) -> list[Forecast]: ...
    def summary(self) -> dict: ...
```

The factory accepts keyword-only `settings: dict` and `seed: int`, and returns the adapter. Factory settings are validated by the adapter; unknown settings fail. The runner supplies alpha queries explicitly. FitPartition target eligibility preserves the existing protocol, including the documented difference between full-train GARCH fitting and eligible MLP training examples.

Mandatory factory/adapter metadata is a `ModelSpec` dataclass: `model_id`, `contract_version=1`, `minimum_history` (positive integer), `alpha_min`, `alpha_max` (inclusive), `uses_validation` (bool), `replay_mode` (`"none"` or `"checkpoint"`). The instance exposes it as `spec`. Use one-dimensional finite floating return arrays, integer target indices in [1, len(returns)-1], unique ascending labels/indices, and equal array/label lengths. Empty eligible fitting targets fail for a supervised adapter. Built-in prediction minimum histories are `max(window + 1, 63)` for historical simulation and 63 for GARCH/MLP, preserving current first-origin eligibility; fitting minimums remain estimator-specific. The runner validates queried alphas against both the toolkit domain [0.005, 0.20] and the adapter domain, rejects duplicates/nonfinite levels, and preserves query order. Fewer supported alphas require a separately declared comparison; never silently shrink the grid for one method.

The built-in factory receives its alpha grid and existing budget through resolved settings; the MLP uses that grid during fitting. The custom adapter author declares any training-alpha settings explicitly. `summary()` supplies actual fitting/eligible sample counts, effective fitting boundaries, preprocessing description, settings, seed, selection rule, runtime, parameter count when meaningful (otherwise null plus reason), and external data/pretraining disclosure. The manifest distinguishes author declarations from measured facts.

Optional checkpoint capability: `save_checkpoint(directory: Path) -> None` plus a factory-module `load_checkpoint(directory: Path) -> TailRiskModel`. The selected checkpoint stores fitted preprocessing and parameters; prediction reconstructs state from the supplied history. No pickled executable objects in the worked example. Preserve the MLP's existing JSON/NPZ checkpoint representation through its wrapper. Checkpoint replay never calls `fit`. Built-in historical/GARCH verification retains its independently implemented reference/refit checks; do not substitute generic self-replay for those checks.

### 2.2 Timing, fitting and state rules

- Split by target date. Training labels are observed by the training cutoff. Validation may select a checkpoint when `uses_validation=true`; otherwise pass `None`. Never pass test data to fitting. Fit preprocessing on training only.
- Validation-selected forecasts on that same validation period are development evidence. Train-period scores are in-sample diagnostics. Neither is presented as a sequence of genuinely out-of-sample forecasts from the model's fitting period.
- PredictionInput contains no target values, targets container, oracle, latent sigma, full panel, file handle or future-backed view. The runner joins realized returns after prediction. Calendar target labels are owned by the runner.
- `predict_one` is functionally repeatable for a fixed fitted model, history and alpha query; it must not fit parameters or advance a hidden stream across calls. Each prediction uses the supplied history. Initially recompute filtered state from that prefix; defer cache/batch optimization. This makes out-of-order replay and independent synthetic paths unambiguous.
- Synthetic path histories reset at each partition. Real validation retains the exact prior context supplied by the current panel contract. Do not change GARCH initialization or silently prepend extra history during migration.
- Use existing six-feature math for the built-in MLP. Add a target-free feature-at-origin helper if needed, sharing numerical formulas with the current feature builder. Never append a fabricated future return just to satisfy an old API.
- Custom models may choose their own causal feature shape and normalization. External feeds, pretrained information and different tuning budgets must be disclosed; they are not automatically matched comparisons. No runner network downloads during fitting/prediction.
- Arbitrary Python plugins are trusted local code, not sandboxed. Restricted arguments and fixture checks cannot prove absence of hidden file access, contaminated pretraining, or researcher selection. Say which checks ran, never "100% no leakage".

### 2.3 Outputs, eligibility and failures

Return exactly one Forecast per requested alpha, in query order. The runner owns all IDs, origin/target mapping and realized returns, and emits the existing `forecasts.csv` columns. It validates finite values and `es <= var < 0` for supported successful outputs. Do not swap signs, sort predictions across alpha, clip values or repair units silently. A negative VaR requirement is our supported domain, not a universal distributional theorem.

Keep valid-shape but nonfinite/out-of-domain returned numbers in the saved forecast row so the existing evaluator counts invalid outputs; add a keyed failure explanation. Explicit model failures carry NaN forecasts and `status="failed"`. Missing/duplicate alpha results, malformed shapes or unexpected adapter exceptions stop the run with a diagnostic and partial-run status; they must not produce a successful comparison. Expected per-origin numerical failures use explicit failed results. A fitting failure stops that run; do not remove a model and rerun survivors automatically.

Write `failures.csv` for every new adapter-contract run, even if empty. For contract-v1 runs the keyed reason table covers the union of explicit failed rows and invalid successful outputs, with exactly one reason row per affected key; legacy tables retain their previous semantics. For explicit failures require a nonempty reason; invalid-output reasons use the runner's vocabulary `non_finite_output` / `outside_supported_domain`. Generic plugin reasons are author-reported, not independently established. Retain the stronger built-in MLP reconstructed-feature reason checks. Missing optional scale is serialized as NaN and is not itself an invalid forecast for custom models; built-in scale checks remain capability-specific.

Derive expected forecast keys from inputs, configured models, permitted origins, each minimum-history requirement and alpha grid, not from the produced rows. Runner eligibility is determined before model execution from the permitted origin set and that adapter's declared minimum history; adapters must not choose their own scored subset. Origins excluded by declared warm-up are counted separately; failures after warm-up stay visible. Preserve existing built-in first-origin rules in compatibility tests even where the historical implementation is more conservative than the mathematical minimum. Do not shorten every model's fitting data to the largest lookback.

All comparisons use the evaluator's existing common valid-row intersection per series/split/alpha, with eligible, failed, invalid, matched and dropped counts. Never report a model ranking without the loss of coverage. Cross-alpha crossing is a diagnostic; it does not remove otherwise valid rows. No new pairwise-specific masks in Stage 2.

### 2.4 Configuration and compatibility

Preserve `schema_version: 1`, existing model IDs/configurations, CLI commands and forecast columns. Mark new manifests with additive `adapter_contract_version: 1`; verifiers dispatch on its presence so legacy results do not acquire new missing-artifact failures. Activate this manifest marker together with its required artifact writer/verifier in S2.3; S2.2 parity runs retain legacy artifact requirements. Preserve all existing legacy integrity failures. New fields are additive and documented; do not rewrite historical configs/runs.

Custom factory configuration is an optional top-level mapping, alongside the existing `models` list:

```json
"models": ["historical_simulation", "garch_student_t", "example_empirical"],
"custom_models": {
  "example_empirical": {
    "factory": "examples.custom_model.model:create_model",
    "settings": {"window": 1000},
    "seed": 101
  }
}
```

Resolve only explicitly named importable `module:function` factories. Reject built-in name collisions, undeclared selected IDs, unused custom entries, duplicate models and invalid settings. No filesystem discovery, `eval`, package installation or network resolution. Users install their own model dependencies. Document that importing a factory executes trusted Python. Record factory identifier, local module source hash when available, package version, declared dependencies and checkpoint file hashes. Missing provenance stays explicit; hashing one module does not prove complete dependency provenance.

New-run verification retains existing artifact checks, adds generic adapter completeness/timing/provenance checks, and uses replay only where implemented. Missing promised checkpoints or replay mismatches are failures. `replay_mode="none"` is unavailable coverage, never a replay pass. Add `verification_coverage.json` listing checks with `passed`, `failed`, `unavailable`, or `not_applicable` and reasons; derive it from actual check results/capabilities, not an independent second verifier. Legacy boolean checks remain readable. Required integrity failure gives `verify` a nonzero exit; unavailable optional replay alone need not, but must label coverage partial in report and terminal output. No single "model approved" result.

## 3. CSV ingestion contract

Extend the existing `ingest --config ... --out ...` command, not a parallel data pipeline. A local source is:

```json
"source": {
  "kind": "returns_csv",
  "path": "../examples/returns.csv",
  "series_id": "example",
  "provenance": "Locally generated example; see generator and seed",
  "session_policy": "observed_sessions",
  "gap_policy": "reject_unresolved"
},
"units": "decimal_log"
```

This fragment lives in the existing ingestion configuration with explicit cutoffs, `context_returns`, `required_context`, alpha grid and model settings. Resolve this source path relative to the ingestion config file. Do not silently change legacy path-resolution semantics; document both. New local CSV mode requires exactly `session_date,return`, ISO YYYY-MM-DD dates, ascending unique dates, finite numeric returns, one series, and units `decimal_log`. No automatic sorting, duplicate removal, return-type inference or OHLCV requirement. A date t labels the return ending at t; the author declares that it spans the preceding actual session. A return file cannot itself prove session completeness.

Reuse `normalise_returns`, `build_panel`, audits/hashing, and chronological checks. No network calls in local mode. Existing provider configs remain valid. Session-calendar checks, if available, are evidence rather than automatic reindexing. Use the existing long-gap detector; require explicit dated resolutions for flagged gaps in `source.gap_resolutions` (mapping return-ending date to nonempty explanation) and retain them as user declarations in the audit. Unresolved suspicious gaps or confirmed missing trading observations stop ingestion in this first version; do not add automatic repairs/resets or delete stress dates. Public docs state the limits of observed-session validation.

## 4. Report contract

`evaluate` writes numerical derived artifacts; `report` only renders saved artifacts and plots from them. It does not fit, download, select models or query an LLM. Extract existing Markdown rendering into `reporting.py`; preserve current CLI behavior. Markdown is primary, `report.json` is the agent-readable equivalent with schema version 1. Links inside reports are relative to the run directory.

Required sections, in order:

1. **Experiment:** question, data/provenance, units, horizon, actual splits, exposure/selection status, seeds, runtime and versions.
2. **Comparison conditions:** model history, actual fit counts, information/pretraining declarations, preprocessing, tuning/checkpoint rules and parameter count when meaningful. Label whole-method versus controlled architecture comparison; default whole-method. For a controlled claim, list matching evidence, not merely a config label.
3. **Integrity and verification coverage:** required checks, replay availability, unresolved findings, failures and exclusions. No statistical p-value contributes to an integrity pass.
4. **Performance by alpha:** existing metrics and common-mask counts; paired mean FZ0 differences versus each included conventional baseline. Lower FZ0 is better; preserve each seed/run rather than treating seeds as new market samples.
5. **Exceedance diagnostics:** observed/expected counts, rate, existing Kupiec/Christoffersen outputs, transition counts and chain breaks. Undefined tests stay undefined. Show sparse counts and inference limitations without arbitrary refusal thresholds. No ES-specific calibration claim.
6. **Risk-curve diagnostics:** within-alpha admissibility separately from adjacent-alpha VaR and ES crossing counts on complete per-origin grids. Count incomplete grids. A grid check is not a continuous monotonicity or integrated-quantile identity proof.
7. **Synthetic accuracy:** existing oracle errors only when an independent oracle exists; otherwise not applicable. Oracles never enter ordinary fit/predict inputs.
8. **Interpretation limits and artifact links:** development selection, sample dependence, missing replay, failures, scope, provenance and exact rerun commands.

Keep `metrics.csv` columns stable; write additive `score_decomposition.csv`, `paired_scores.csv`, and `curve_diagnostics.csv`. Decomposition uses exactly the common scored rows: `shortfall = -max(v-y,0)/(alpha*e)`, `ratio = v/e - 1`, `log = log(-e)`; their means must sum to mean FZ0 within stated float64 tolerance. Paired scores use that same common mask, with mean difference, n and comparator identity, not significance stars or confidence intervals. No capital-cost, regulatory-suitability, head-causality, or clipping-bias interpretation.

Three fixed optional PNGs under `plots/`, using Matplotlib Agg: (a) realized returns and VaR/ES with exceedance markers at primary alpha .025, (b) coverage rate versus alpha with nominal reference, (c) last common complete evaluation origin's VaR/ES curves versus alpha, with date and units. If .025 is absent, omit (a) with a reason rather than choose a favorable alpha. Last-origin curves are descriptive, not representative or live forecasts. No interactive dashboard or automated crisis labeling. `report --plots` requests plots and fails clearly with the installation command when Matplotlib is absent; ordinary report works without it.

## 5. Ordered implementation slices

Each slice is one reviewable milestone, not necessarily one commit. Run only relevant tests plus the designated small smoke. Do not rerun the nine-run development experiment. Update the brief decision/status logs where present. Stop after the requested slice with changed files, exact checks, limitations and next slice; continue to another only when the task authorizes it. A passing gate never authorizes a Git mutation or publication.

### S2.1 — Public protocol and target-free inputs

**Deliver:** `models/protocol.py`, runner input-construction helpers, and focused tests. Add target-free feature-at-origin computation sharing existing math; do not change estimator fitting or route production runs yet.

**Steps:** implement dataclasses/validation; convert existing contexts into owning causal prefixes and fitting partitions; map target indices without changing eligibility; implement named Forecast validation; document contracts in module docstrings.

**Gate:** assert exact prefix/labels at first, middle and last origins; no target/oracle/full-array/base-view access; mutation does not affect parent arrays; fit cutoff checks; independent paths reset; existing and target-free six features/scales match; malformed alpha/result fixtures fail. All existing feature/timing tests still pass. No training experiments.

### S2.2 — Built-in adapters and runner migration

**Deliver:** `models/adapters.py`, `runner.py`, built-in registry and CLI delegation using public contract; legacy wrappers remain supported.

**Steps:** wrap historical numerical function; use fitted GARCH parameters and the exact existing recursion/initialization through the origin; use existing MLP trainer/checkpoint and target-free inference; retain existing split/context/first-origin masks and failures; emit existing rows and new coverage metadata. Do not create dummy future returns to call legacy prediction APIs.

**Gate:** small deterministic old/new forecasts match in keys, signs, scales and numbers within declared precision tolerances (conventional float64 rtol 1e-10/atol 1e-12; MLP dtype-aware, recorded). Include Gaussian/t paths and a dated local panel fixture, with no real-data refitting. Verify input arrays actually handed to prediction. Changing evaluation futures with fitted parameters fixed cannot change earlier forecasts. Do not perturb validation then retrain and call checkpoint-selection changes leakage. Repeated/out-of-order calls agree. Existing replay/completeness/failure and end-to-end tests pass. Capture old-route references before removing dispatch.

### S2.3 — Bring-your-data and bring-your-model

**Deliver:** local CSV ingest mode; custom factory resolution; generic verification coverage; `examples/custom_model/model.py` plus configuration; initial `docs/DATA_CONTRACT.md` and `docs/MODEL_ADAPTER.md` matching implemented APIs.

**Steps:** implement source contract; add explicit factory mapping; implement a tiny rolling empirical example reusing the existing historical quantile math, clearly labeled an integration example rather than a new benchmark. Give the example JSON checkpoint support containing its settings. Generalize expected-key checks, missing-scale handling and failure records; preserve legacy verifier paths and MLP-specific reason checks. Store the scope of unverifiable plugin declarations.

**Gate:** run example beside built-ins on a local synthetic CSV; identical sample keys, no evaluator source edits for registration, factory/ID/units/date/gap errors clear. Tests detect deliberately missing origins/alphas, wrong target mapping and invalid outputs. Assert the plugin cannot obtain a future target through supplied arguments; do not claim prevention of external file access. Verify checkpoint replay with fit disabled, changed/deleted checkpoint fails, and a no-replay fixture yields partial coverage rather than a fake pass. A malformed factory stops before comparison. Imported-data source hashes/cutoffs replay. Ordinary CSVs never acquire oracle ground truth merely because they were originally simulated.

### S2.4 — Evidence report and plots

**Deliver:** report contract above, derived tables, `report.json`, optional plotting extra, concise interpretation docs.

**Steps:** centralize mask reuse; add decomposition, paired differences and crossing counts; extract rendering; preserve legacy artifacts; render plots from saved rows only. Mark missing optional legacy metadata unavailable, never manufacture provenance. Link every claim to a table/check.

**Gate:** hand-computable loss-decomposition fixture, known crossing/incomplete-grid fixture, same-mask paired difference, invalid/failed-row accounting, undefined coverage test, and real/no-oracle report. Report values agree with saved metrics. Inspect generated Markdown/JSON and all three PNGs on a small offline run. Rendering performs no fit/download. Missing plotting dependency gives actionable feedback. No hard-coded check totals or prohibited claims.

### S2.5 — Public packaging and runnable examples

**Deliver:** self-contained README; completed data/model docs; `docs/METHODS.md` with methods, attribution and limitations; `examples/README.md`; two explicit synthetic configs; regular `skills/tail-risk-audit/SKILL.md` with references to the public docs; tested environment record. No dependency on private planning documents.

**Fixed example choices:** preserve historical configs. Add `configs/examples/student_t.json` and `gaussian.json` using existing DGP parameters, burn-in 2000, train 3000/validation 1500, W=1000, current six alphas, baselines only, explicit train/validation execution, no test partition. Use data seeds 6101/6102 for Student-t and 6201/6202 for Gaussian. These are execution/reuse examples with limited tail counts, not new performance evidence. Generate the small returns CSV locally from a documented command; do not bundle market snapshots. Include an opt-in MLP command using existing settings without tuning; the primary quickstart does not require JAX.

**Dependencies:** add `baselines` extra for `arch` and `plots` extra for Matplotlib; retain existing `dev` compatibility and `neural` optional imports. Record a tested environment and measured runtime, not a promised 15-second limit. Update metadata description to research-toolkit purpose. Do not select a project license for the user: prepare attribution and flag the license decision as a release prerequisite, without blocking independent implementation.

**Skill:** regular skill, matching directory/name `tail-risk-audit`, YAML name/description, one workflow (inspect inputs -> resolve config -> run -> verify -> evaluate -> report -> explain evidence/failures). No mandatory LLM inside numerical execution. Use a documented working directory and installed CLI; resolve reference paths relative to the skill. Model integration instructions belong in linked public docs, not a second orchestration system. Include the required reference pages under the skill's `references/` at packaging time with a check that they match the canonical public docs, so a copied skill does not depend on `../../docs`; the engine still requires the documented package installation. Follow installed Jiuwen help; do not invent frontmatter `inputs`/`outputs` schemas or require Team Skill validators for a regular skill.

**Gate:** fresh virtual environment install and full quickstart succeeds; documented custom example import works from the stated directory; all local public links resolve; optional dependencies genuinely optional; no private paths/secrets/restricted data in proposed public tree. Skill loads in actual WorkSwarm; loading alone is not execution evidence. README distinguishes available features from remaining gates.

### S2.6 — Fresh WorkSwarm execution and course handoff

**Deliver:** compact `evidence/stage2/` execution record, sanitized 2–4 screenshots or equivalent records, output/check references, report/demo outline and remaining release checklist. Do not bulk-commit generated runs or chats.

**Case:** after the primary Student-t quickstart, use Gaussian config from S2.5 in a fresh WorkSwarm conversation and new output folder. Show planned inputs and follow the assignment's fresh-run approval procedure, then actually execute run / verify / evaluate / report. This is changed-case self-reproduction, not a new independent statistical claim. A second real index or jump DGP is not required.

**Gate:** record installed runtime version, loaded skill source, exact commands, exits, checks, produced files, measured runtime and one inspected financial calculation. Evidence must establish actual WorkSwarm use, not just Codex execution. If runtime access is unavailable, report that specific remaining gate; do not claim completion or delay unrelated deliverables.

Course handoff additionally tracks: public GitHub URL and exact final commit, README openJiuwen link, matching source ZIP, report PDF at least three pages excluding references/short appendices, five-minute demo, and attribution. License selection, publication permission, LMS deadline and live/recorded demo instructions are human/external prerequisites when still unresolved. Hub acceptance, upstream PRs, novel models and model superiority are not Stage 2 gates.

## 6. Definition of done and next-stage boundary

A new user can install, run a synthetic case, supply the documented CSV, integrate the worked adapter without changing evaluator internals, and inspect both results and limits of verification. Existing reviewed evidence remains reproducible under its recorded environment. A failed statistical diagnostic can be a valid experiment; a broken interface cannot.

After S2.6, report Stage 2 evidence and remaining publication/course actions. Do not automatically begin M0, CAESar, new real-data runs, tuning, or final-test scoring. Stage 3 research remains available through the adapter and requires a separate bounded implementation instruction. Continue independent mathematical research in its own research area without making it a prerequisite of reusable delivery.
