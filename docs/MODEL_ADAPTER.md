# Model adapter contract (v1)

How to add a model to `tailrisk` without editing the evaluator, the runner or the verifier.
The contract is fixed by `src/tailrisk/models/protocol.py`; registration is implemented in
`src/tailrisk/models/registry.py`; the worked example is `examples/custom_model/model.py`.
If this page and the code disagree, the code is right.

## The two files you write

1. a Python module with a factory function and an adapter class;
2. a `custom_models` entry in the run configuration naming it.

Nothing else. No entry points, no installation, no filesystem discovery, no `eval`.

## The factory

```python
def create_model(*, settings: dict, seed: int | None):
    return MyAdapter(settings=settings, seed=seed)
```

* Called with **keyword arguments only**; `settings` is your `custom_models` entry's object,
  `seed` its integer (or `None`).
* The factory receives **no alpha grid and no data**. A model that trains at specific alpha
  levels must declare them inside its own `settings`.
* Unknown settings are your responsibility: validate and raise. An ignored typo silently
  runs a different model than the one the config describes.
* Exceptions from your factory are reported as `ContractError` with your message attached.

## The adapter

```python
class MyAdapter:
    spec: ModelSpec          # required, set at construction

    def fit(self, train: FitPartition,
            validation: FitPartition | None = None) -> None: ...
    def predict_one(self, inputs: PredictionInput,
                    alphas: np.ndarray) -> list[Forecast]: ...
    def summary(self) -> dict: ...
```

`ModelSpec` fields (dataclass, frozen):

| Field | Rule |
|---|---|
| `model_id` | must equal the id in `custom_models`; a mismatch is refused at construction |
| `contract_version` | `1` |
| `minimum_history` | positive int; the warm-up the runner must honour before the first origin |
| `alpha_min`, `alpha_max` | inclusive adapter domain inside the toolkit's `[0.005, 0.20)` |
| `uses_validation` | `True` only if `fit` genuinely selects on the validation partition |
| `replay_mode` | `"none"` or `"checkpoint"` |

`predict_one` returns **exactly one `Forecast` per requested alpha, in query order**. Missing,
duplicate or reordered alphas stop the run with a diagnostic instead of producing a
comparison.

### What you are handed

`PredictionInput` has exactly five fields: `series_id`, `path_id`, `origin_label`,
`history`, `history_labels`. `history` ends at the origin — its last element **is** `r_t` —
and it is an owning, read-only copy, so there is no `.base` view onto later returns. There is
no target, no realized return, no oracle, no panel and no file handle. Reading the future
from the arguments you are given is not possible; reading it from somewhere else in your own
code is outside what any argument-level check can prevent, which is why the trust statement
below exists.

### What the runner owns

Row identity, the origin→target mapping, realized returns, split boundaries, the alpha grid,
failures accounting and the artifact layout. An adapter never chooses which origins are
scored: eligibility is `permitted_origins = allowed_origins ∩ [minimum_history - 1, n - 2]`,
computed before your code runs. Origins inside your declared warm-up are counted as excluded
warm-up; a permitted origin you cannot answer must still produce a row (`status="failed"`).

### Reporting failures

* **Expected per-origin numerical failure**: return `Forecast(..., status="failed",
  reason="<short, stable name>")` with NaN numbers. The reason must be nonempty.
* **Out-of-domain but finite** (`es > var`, or `var >= 0`, or NaN/infinite numbers): return
  the numbers with `status="ok"`. The row is kept, the evaluator counts it as an invalid
  output, and the verifier enters it in `failures.csv` with `non_finite_output` or
  `outside_supported_domain`.
* **A bug** (wrong shape, wrong count, unexpected exception): stop the run. A crashed adapter
  must not look like a successful comparison.
* `scale` is an optional causal diagnostic. `None` serializes as NaN and is **not** an
  invalid forecast for a custom model.

## Registration

```json
{
  "models": ["historical_simulation", "garch_student_t", "example_empirical"],
  "custom_models": {
    "example_empirical": {
      "factory": "examples.custom_model.model:create_model",
      "settings": {"window": 60},
      "seed": 101,
      "dependencies": []
    }
  }
}
```

| Entry key | Meaning |
|---|---|
| `factory` | `"module:function"`, resolved by import — nothing else is searched |
| `settings` | passed through to your factory as a dict |
| `seed` | passed as `seed`; non-negative int or absent |
| `dependencies` | author-declared package names; recorded, **not** installed or verified |

Refused, with a message naming the problem: a custom id colliding with a built-in id; a
configured entry that is not in `models` (a stale claim about what the experiment contained);
a selected model with no declaration; a repeated id; a malformed `module:function`; unknown
`custom_models` or entry keys; a factory that returns an object without a matching `spec`.

The factory is resolved and constructed **before any data is read**, so a malformed
declaration fails without producing artifacts.

`module:function` resolves from the working directory. Run from the repository root for the
bundled example; `examples/` is a regular package for exactly that reason.

## Checkpoints (optional capability)

If your state cannot be reproduced from settings alone, declare `replay_mode="checkpoint"`
and provide all three:

```python
def save_checkpoint(self, directory: Path) -> None:        # on the adapter
def checkpoint_metadata(self) -> dict:                     # on the adapter, optional but
                                                           #   required for hashes
def load_checkpoint(directory: Path):                      # module-level, required
```

* Both `save_checkpoint` and `load_checkpoint` receive the **run directory**; your layout
  inside it is yours. The example uses `checkpoints/example_empirical/checkpoint.json`.
* `checkpoint_metadata()` returns `{"path": "<relative dir>", "files": {"<name>": {"sha256": ..., "bytes": ...}}}`.
  The run records hashes computed from the bytes on disk, not from this self-report.
* `load_checkpoint` must rebuild an inference-ready adapter **without calling `fit`** and
  without any training data. No pickled executable objects: JSON and arrays only, in the
  example.
* A `replay_mode="checkpoint"` adapter whose module has no `load_checkpoint` is refused at
  registration: a checkpoint nobody can load is not reusable state.

## What the run records

In `manifest.json`, additively to `schema_version: 1`:

| Key | Content |
|---|---|
| `adapter_contract_version` | `1` for every run written by this version |
| `custom_factories` | factory string, module file, module SHA-256, distribution + version (or explicit `null`), declared dependencies, trust note |
| `checkpoints` | per model: relative path and `{sha256, bytes}` for each promised file, hashed from disk |
| `model_coverage` | per model: `replay_mode`, `minimum_history`, per-split origin/ok/failed/invalid counts |

Missing provenance stays missing (`null` with a note) rather than being filled in.

## What `verify` checks

Generic, for every executed model: expected keys derived from your declared
`minimum_history`, the configured splits and the alpha grid — not from the rows you produced;
the signed domain `es <= var < 0`; and the keyed reason table.

Custom-specific:

| Check | Failure means |
|---|---|
| `custom_factory_provenance` | the manifest does not record where the code came from |
| `custom_factory_source:<id>` | the factory importable now is not the factory the manifest recorded: a different factory reference, or a different SHA-256 for the module file |
| `model_capabilities_match:<id>` | the recorded `minimum_history` or `replay_mode` disagrees with the specification reconstructed from the code |
| `forecast_keys_complete:<id>` | a permitted origin or alpha is missing, unexpected or duplicated |
| `failure_reasons_cover_affected_keys` | a failed row or invalid output has no reason row, has two, or its reason disagrees with the row |
| `checkpoint_files_match_manifest:<id>` | a promised file is deleted, replaced, or was never written |
| `checkpoint_replay:<id>` | replaying the saved state from causal histories disagrees with the saved rows, or the replay does not answer the full query |

Replay tolerance is `rtol=1e-10, atol=1e-12` on `var`/`es`/`scale`, and status must agree.
Each replay response is validated against the query — exactly one `Forecast` per requested
alpha, in query order — so a shortened, duplicated, extra or reordered answer fails instead
of being compared partially. The check also requires `requested == answered` and
`compared + failed_rows == requested`, so "the loop ran" is never mistaken for "the query was
covered".

`custom_factory_source` is a comparison, not a record. It imports the module (the same one
replay executes), hashes the file those imports came from *now*, and compares that with the
manifest. Two consequences are worth stating plainly: an edit made after a run invalidates
the run even if the saved numbers still reproduce; and hashing establishes what the file
contains now, not that the already-imported bytecode was re-read (Python caches imports). If
the run recorded no hash, or the module has no readable source, the check is reported as
**unavailable** — an absence of evidence, never a pass.

`failure_reasons_cover_affected_keys` keys on `(model_id, series_id, origin_date, alpha)`:
`failures.csv` is one table for the whole run, so two models documenting the same origin are
two rows, not a duplicate. The reason text is held to two different standards. An invalid
output — a successful row outside `es <= var < 0` or with a non-finite number — must carry
exactly the reason recomputed from its own numbers, because free text there would disagree
with the evidence. An explicit `status="failed"` row is the adapter's own declaration, so its
text is recorded but not independently established; it must merely be nonempty, and a missing
or `NaN` entry counts as empty rather than as the string `"nan"`.
A model that declares `replay_mode="none"` is reported as **unavailable** coverage: `verify`
still exits zero if everything else holds, but `verify` prints that coverage is partial and
`verification_coverage.json` records the gap. Unavailable is never a pass.

## What none of this establishes

* The factory is arbitrary trusted local Python. It is not sandboxed. Importing it executes
  its import-time code, and nothing here inspects what it reads at prediction time.
* Hashing one module says nothing about its dependency tree, and nothing about whether a
  checkpoint was pretrained somewhere this run did not observe.
* `dependencies` is an author declaration, not an installation or a resolution.
* A verifier cannot distinguish "reproducible from a checkpoint" from "reproducible from a
  checkpoint that was fitted on data the author should not have used". Fitting-boundary
  checks constrain the *caller*; they cannot audit the author's history.
* Where a failure reason belongs to an explicit `status="failed"` row, it is recorded, not
  independently established. (An *invalid output* is different: its reason must equal the one
  the verifier recomputes, so it is checked rather than recorded.)
* `custom_factory_source` compares one file's bytes. It does not re-import in a clean process,
  so it cannot detect an edit that was made and reverted before `verify` ran, and it says
  nothing about `sitecustomize`, import hooks, or a module that resolves elsewhere on a
  different `sys.path`.

Say which checks ran. Never "no leakage".

## Worked example

`examples/custom_model/model.py` is a rolling empirical VaR/ES that reuses the shipped
`tailrisk.historical.historical_var_es` — an *integration example*, not a new benchmark and
not a new estimator. It shows the smallest complete plugin: factory, spec, causal prediction,
explicit failure rows, and a JSON checkpoint that stores settings and no observations.

```bash
python examples/custom_model/make_demo_returns.py

PYTHONPATH=src python -m tailrisk.cli ingest \
  --config examples/custom_model/ingest.json --out outputs/example_local/dataset

PYTHONPATH=src python -m tailrisk.cli run \
  --config examples/custom_model/run.json --output outputs/example_local/run

PYTHONPATH=src python -m tailrisk.cli verify --run outputs/example_local/run
PYTHONPATH=src python -m tailrisk.cli evaluate --run outputs/example_local/run
PYTHONPATH=src python -m tailrisk.cli report --run outputs/example_local/run
```

`verify` re-derives the series from the panel, replays every saved forecast, and confirms the
checkpoint by hash and by content. Delete or edit
`outputs/example_local/run/checkpoints/example_empirical/checkpoint.json` and re-run `verify`
to see integrity fail; the tamper checks are asserted permanently in
`tests/test_custom_model.py`.
