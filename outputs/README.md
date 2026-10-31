# Local run artifacts

Generated runs are ignored by Git. Use a new directory for each run; the current CLI can overwrite an existing directory.

- `smoke/` is preserved from before the timing fix. Its `SUPERSEDED.md` explains why it must not be used for evaluation.
- Migration validation outputs, if present, are synthetic execution checks only.

Do not edit historical manifests to make old outputs look current. Curate small lawful examples separately when preparing a release.
