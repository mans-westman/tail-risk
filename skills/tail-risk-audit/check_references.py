"""Check that the skill's reference pages are the canonical public docs.

A copied skill must not depend on ``../../docs``: the pages under
``skills/tail-risk-audit/references/`` are byte-identical copies of the repository's public
pages. Copies drift, so this script compares them by SHA-256 and, on a mismatch, says which
canonical file to re-copy -- it never rewrites anything itself.

Usage, from the repository root::

    python skills/tail-risk-audit/check_references.py

Exit status is 0 when every copy matches and 1 otherwise, so it can gate a release or a
commit without being wired into the package's test suite (which also has a parity test).
Run from an installed copy, where no ``docs/`` sits alongside the skill, there is nothing to
compare: the script says so and exits 0 instead of reporting five missing pages.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = REPO_ROOT / "docs"
REFERENCE_DIR = SKILL_DIR / "references"

#: Page stem -> canonical file name. The skill copies these five and no others.
PAGES = (
    "METHODS.md",
    "DATA_CONTRACT.md",
    "MODEL_ADAPTER.md",
    "REPORTING.md",
    "ENVIRONMENT.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def check() -> list[str]:
    """Return one human-readable line per problem; an empty list means parity."""
    problems: list[str] = []
    for name in PAGES:
        canonical = CANONICAL_DIR / name
        reference = REFERENCE_DIR / name
        if not canonical.exists():
            problems.append(f"{name}: canonical page {canonical} is missing")
            continue
        if not reference.exists():
            problems.append(
                f"{name}: reference missing; copy {canonical} to {reference}"
            )
            continue
        if _sha256(canonical) != _sha256(reference):
            problems.append(
                f"{name}: reference differs from {canonical}; re-copy it (cp docs/{name} "
                f"skills/tail-risk-audit/references/{name})"
            )
    extra = sorted(
        path.name
        for path in REFERENCE_DIR.glob("*")
        if path.is_file() and path.name not in PAGES
    )
    for name in extra:
        problems.append(
            f"{name}: reference has no canonical page in docs/; remove it or add the page"
        )
    return problems


def main() -> int:
    if not CANONICAL_DIR.is_dir():
        print(
            f"no canonical docs/ at {CANONICAL_DIR}; this skill copy is installed outside the "
            "repository, so there is nothing to compare"
        )
        return 0
    problems = check()
    if problems:
        print(f"reference check FAILED ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"reference check passed: {len(PAGES)} pages identical to docs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
