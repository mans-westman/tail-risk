# Data contract: a local returns CSV

This page describes the one data source this stage adds: a **declared local returns file**.
It states exactly what the importer accepts, what it refuses, what it records, and — in its
own section — what it does *not* and cannot verify. It matches the implementation in
`src/tailrisk/data.py`; where the two ever disagree, the code is right and this page is a bug.

A returns file is a *declared input*, not a provider feed. That distinction drives every
decision below: nothing is inferred from magnitude, nothing is repaired, and nothing is
sorted.

## The file

| Requirement | Rule |
|---|---|
| Columns | exactly `session_date,return` — no extras, no missing |
| Dates | ISO `YYYY-MM-DD`, valid calendar dates |
| Order | strictly ascending **as written**; the importer never sorts |
| Uniqueness | no repeated session date |
| Returns | numeric, finite; decimal log-return units |
| Rows | at least one; the series must not be constant |
| Series | exactly one series per file |

```
session_date,return
2021-01-04,-0.000010946876
2021-01-05,0.008845500686
...
```

A date `t` labels the return **ending** at `t`, spanning the previous observed session. The
file's author declares that; see the limits section.

## The ingestion configuration

```json
{
  "schema_version": 1,
  "dataset_id": "example_local",
  "source": {
    "kind": "returns_csv",
    "path": "data/returns.csv",
    "series_id": "example_local",
    "provenance": "Locally generated; regenerate with make_demo_returns.py",
    "session_policy": "observed_sessions",
    "gap_policy": "reject_unresolved",
    "gap_resolutions": {}
  },
  "units": "decimal_log",
  "cutoffs": {"train_end": "2022-01-03", "validation_end": "2022-04-25"},
  "context_returns": 101,
  "required_context": 100,
  "historical_window": 100,
  "alpha_grid": [0.005, 0.01, 0.025, 0.05, 0.1, 0.2],
  "exposure": "free text; recorded, not verified"
}
```

### `source`

| Key | Required | Meaning |
|---|---|---|
| `kind` | yes | `"returns_csv"`. The other supported kind is `"yahoo_chart"` |
| `path` | yes | **relative to the ingestion config file**, not to the working directory |
| `series_id` | yes | the series identifier written into the audit |
| `provenance` | yes | where the bytes came from; recorded verbatim |
| `session_policy` | yes | `"observed_sessions"` is the only supported value |
| `gap_policy` | yes | `"reject_unresolved"` is the only supported value |
| `gap_resolutions` | no | `{"YYYY-MM-DD": "nonempty explanation"}` for flagged gaps |

Unknown keys are an error, per source kind. A provider config (`"kind": "yahoo_chart"`) keeps
its own key set and is unaffected: a key that the chosen kind cannot honour is refused
rather than ignored.

### Path resolution, stated twice because it differs

* `source.path` inside an **ingestion** config resolves relative to that config file.
* `data.panel` / `data.audit` inside a **run** config resolve relative to the working
  directory, exactly as they always have.

The example commands below are written to be run from the repository root, so the run config
points at `outputs/example_local/dataset/panel.csv`.

### Units

`units` must be `decimal_log`. Percent inputs are refused rather than converted:

```
local returns CSV requires units 'decimal_log', got 'percent_log';
convert the file rather than letting the importer guess
```

### Gaps

The existing long-gap detector flags any session whose elapsed calendar days since the
previous observed session exceed 4 (so an ordinary long weekend is fine). Every flagged gap
must carry a dated explanation:

* a flagged gap with no explanation stops ingestion, and the message names the date;
* an explanation for a date that is not flagged is refused as a stale claim;
* a blank explanation is refused.

Nothing is forward-filled, reindexed, interpolated or dropped. The explanations are kept in
the audit as **author declarations**, labelled as such.

### Splits and context

`cutoffs`, `context_returns`, `required_context` behave exactly as they do for a provider
snapshot: splits are assigned by target date, and each split carries its preceding returns as
causal context, so the first scored target of a split is not its 64th or 101st session.

`context_returns` must be at least `required_context + 1`: the origin session that predicts a
split's first target needs a complete trailing window. A series too short for that is an
error, not a silent crop.

## What the audit records

`ingest` writes `panel.csv` and `data_audit.json`. For a returns file the audit contains:

| Field | Content |
|---|---|
| `source.raw_sha256` | SHA-256 of the file's bytes — the re-identification handle |
| `source.resolved_path` | the absolute path actually read |
| `source.provenance`, `session_policy`, `gap_policy` | as declared |
| `source.gap_resolutions` | the explanations, retained as declarations |
| `coverage.returns`, `first/last_return_target` | count and observed span |
| `validation.*` | rows read, duplicates (0 by construction), long gaps, unresolved gaps |
| `units.returns` | `decimal_log` |
| `splits` | per-split targets, scored origins, attached context, and the cutoffs used |
| `calendar` | `{"status": "not checked", "reason": ...}` |
| `cross_check` | `null` |
| `limitations` | the four statements below, in machine-readable form |

`oracle_curves.csv` is **never** written for this data, even when the series was originally
simulated: a supplied CSV carries no ground truth, and `dataset_kind` is `real`.

## What this cannot verify

Stated plainly, because a reader will otherwise assume otherwise:

1. **Session completeness.** A returns file cannot prove its sessions are a complete exchange
   record. The `observed_sessions` policy is the author's declaration.
2. **Calendar correctness.** No exchange calendar is consulted. `calendar.status` is
   `not checked`, not `passed`.
3. **Gap meaning.** A resolved gap is an explanation, not a verified closure.
4. **Units and provenance.** Units are declared, never inferred; `provenance` is free text
   recorded verbatim and is not evidence.
5. **Corrections.** If the file was revised after the fact, nothing here can tell.

Choosing which gaps are real is therefore an authoring decision, recorded rather than
laundered. Previewing `data_audit.json` before `run` is the intended workflow.

## Commands

```bash
# 1. generate the example's deterministic input (no RNG, same bytes every time)
python examples/custom_model/make_demo_returns.py

# 2. ingest: validate, build the panel, hash the source
PYTHONPATH=src python -m tailrisk.cli ingest \
  --config examples/custom_model/ingest.json \
  --out outputs/example_local/dataset

# 3. inspect the audit before scoring anything
#    outputs/example_local/dataset/data_audit.json
```

Typical refusals:

```
returns CSV dates are not strictly ascending; no sorting is performed because the file
order is the declared session order (first break at row 7)

returns CSV repeats the session date 2021-02-11

returns CSV carries unexpected column(s): ['close']; this mode accepts exactly
['session_date', 'return']

3 long gap(s) have no dated explanation: ['2021-03-15', ...]; add each to
source.gap_resolutions as return-ending date -> reason, or supply a complete series.
Nothing is filled, reindexed or dropped automatically

source.gap_resolutions explains date(s) that are not flagged gaps: ['2021-03-01'];
a resolution for a date with no gap is a stale claim

local returns CSV ingestion does not accept ['from_raw']; a declared file is read as-is,
with no downloaded payload or reference series to compare against
```
