# Local run artifacts

Generated runs are ignored by Git. Nothing here is part of the release: a clone contains code,
documentation, examples and the curated trial record under `../evidence/stage2/s26/`, not run
outputs. Runs you create stay on your machine.

`run` refuses a nonempty output directory, so every run needs a fresh path; `verify`,
`evaluate` and `report` intentionally overwrite only their own derived artifacts
(`artifact_verification.json`, `metrics.csv`, `report.md`).

## What ships instead

- [`../evidence/stage2/s26/`](../evidence/stage2/s26/03_RESULTS.md) — the fresh-agent WorkSwarm
  trial: exact commands, exit codes, runtimes, results, figures, and the post-trial reporting
  repair with superseded figures labelled.
- [`../docs/ENVIRONMENT.md`](../docs/ENVIRONMENT.md) — tested environments and measured runtimes.
- [`../docs/METHODS.md`](../docs/METHODS.md) — what the numbers do and do not establish.

## Development run record (local, not distributed)

The 2026-10-31 … 2026-11-04 development runs — synthetic smoke runs, the nine-run comparison,
the S&P 500 development runs and the first real-data run — live only in this working copy.
Their per-directory inventory is therefore a local record, not a reader-facing page. Two facts
from it are worth carrying in public, because they limit what any old number can be cited for:

- **The real-data run is an exposed retrospective.** Its 2020–2025 test period has been scored
  and its Kupiec/Christoffersen metrics inspected, so it is not an untouched holdout and no
  superiority claim attaches to it. Its manifest also predates panel hashing and frozen cutoffs;
  the current `verify` replays every row with 0 mismatches and still exits 1 on exactly
  `panel_sha256_matches` and `effective_cutoffs_frozen`. Read that as *cannot prove which panel
  it consumed*, not *numbers wrong*, and do not patch the manifest to silence it.
- **Replay is dependency-bound.** A run reproduces under the versions that produced it; a
  numpy/pandas shift moved the fitted GARCH parameters in the sixth to seventh digit and turned
  every GARCH row into a replay mismatch. `verify` names such drift, so it is not misread as
  data corruption.

## Policy for any run directory kept here

- **A corrected specification is a new run.** Historical manifests and forecast files are never
  edited to make an old output look current.
- **Real market data is never stored here.** Only derived artifacts, and `data/` stays ignored;
  the provider's terms apply to the feed, not to this repository.
- **Exposure is recorded, not hidden.** A run whose test period was scored is labelled as an
  exposed retrospective, not an untouched holdout.
