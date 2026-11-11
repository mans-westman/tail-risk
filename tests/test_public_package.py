"""Checks on the public package: links, skill references, and no private files.

These are packaging checks, not numerical ones. They exist because the public documentation
is the deliverable: a broken link, a skill whose reference copies drifted from ``docs/``, or a
public page that points at a gitignored planning file are all release defects that no
numerical test would notice.

Where the checks can share code with the shipped tooling they do: the reference-parity test
calls :func:`skills/tail-risk-audit/check_references.py`, so the command a maintainer runs and
the test the suite runs cannot disagree about which pages are canonical.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "tail-risk-audit"
DOCS_DIR = REPO_ROOT / "docs"

#: Markdown that ships to a reader. Private planning files are deliberately absent.
PUBLIC_MARKDOWN = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "examples" / "README.md",
    REPO_ROOT / "examples" / "bring_your_model" / "README.md",
    SKILL_DIR / "SKILL.md",
) + tuple(sorted(DOCS_DIR.glob("*.md")))

#: The researcher-facing standalone model must stay independent of the toolkit.
STANDALONE_MODEL = REPO_ROOT / "examples" / "bring_your_model" / "ewma_gaussian.py"

#: Files with paths or names that must not reach a public page.
PRIVATE_TOKENS = (
    "AGENTS.md",
    "STATUS.md",
    "DECISIONS.md",
    "PROJECT_GUIDE",
    "LITERATURE_REVIEW",
    "HANDOVER",
    "EXPERIMENT_PROTOCOL",
    "final-project-instructions",
    "/Users/",
    "../.tmp/",
    "file://",
)

_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def _relative_links(text: str) -> list[str]:
    """Targets of inline Markdown links, without external schemes or pure anchors."""
    targets = []
    for target in _LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target.split("#", 1)[0])
    return [target for target in targets if target]


def _reference_check():
    path = SKILL_DIR / "check_references.py"
    spec = importlib.util.spec_from_file_location("tailrisk_skill_references", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_markdown_set_is_not_empty() -> None:
    for path in PUBLIC_MARKDOWN:
        assert path.is_file(), f"expected public page {path}"


def test_public_relative_links_resolve() -> None:
    missing = []
    for page in PUBLIC_MARKDOWN:
        for target in _relative_links(page.read_text(encoding="utf-8")):
            if not (page.parent / target).resolve().exists():
                missing.append(f"{page.relative_to(REPO_ROOT)} -> {target}")
    assert not missing, "unresolved relative links: " + "; ".join(missing)


def test_public_links_stay_inside_the_repository() -> None:
    """A public page must not depend on a path above the repository root."""
    root = REPO_ROOT.resolve()
    escaping = []
    for page in PUBLIC_MARKDOWN:
        for target in _relative_links(page.read_text(encoding="utf-8")):
            resolved = (page.parent / target).resolve()
            if root not in resolved.parents and resolved != root:
                escaping.append(f"{page.relative_to(REPO_ROOT)} -> {target}")
    assert not escaping, "links leaving the repository: " + "; ".join(escaping)


def test_skill_references_match_canonical_docs() -> None:
    problems = _reference_check().check()
    assert not problems, "skill reference drift: " + "; ".join(problems)


def test_reference_check_tolerates_an_installed_copy(monkeypatch, capsys) -> None:
    """An installed copy has no ``docs/`` beside it; that is not drift and must not fail."""
    module = _reference_check()
    monkeypatch.setattr(module, "CANONICAL_DIR", module.SKILL_DIR / "no-such-docs")
    assert module.main() == 0
    assert "nothing to compare" in capsys.readouterr().out


def test_skill_frontmatter_names_the_directory() -> None:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---"), "SKILL.md must start with YAML frontmatter"
    block = text.split("---", 2)[1]
    fields = {}
    for line in block.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"').strip("'")
    assert fields.get("name") == SKILL_DIR.name
    description = fields.get("description", "")
    assert len(description) > 40, "a skill description must say when to use the skill"
    assert "\n" not in description


def test_skill_declares_its_reference_pages() -> None:
    """Every shipped reference page is listed in SKILL.md, and vice versa."""
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    on_disk = {path.name for path in (SKILL_DIR / "references").glob("*.md")}
    assert on_disk, "the skill ships no reference pages"
    unlisted = sorted(name for name in on_disk if f"references/{name}" not in text)
    assert not unlisted, f"reference pages not mentioned in SKILL.md: {unlisted}"


def test_public_pages_do_not_reference_private_files() -> None:
    offenders = []
    for page in PUBLIC_MARKDOWN:
        text = page.read_text(encoding="utf-8")
        for token in PRIVATE_TOKENS:
            if token in text:
                offenders.append(f"{page.relative_to(REPO_ROOT)} contains {token!r}")
    assert not offenders, "; ".join(offenders)


def test_example_configs_state_they_are_not_performance_evidence() -> None:
    """The synthetic examples must carry their own scope statement."""
    for path in sorted((REPO_ROOT / "configs" / "examples").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert "not performance evidence" in text, f"{path.name} lacks a scope statement"


@pytest.mark.parametrize("extra", ["baselines", "dev", "data", "neural", "plots"])
def test_documented_extras_exist_in_pyproject(extra: str) -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf"^{extra} = \[", text, re.MULTILINE), f"missing extra {extra!r}"


def test_standalone_model_imports_only_numpy_and_scipy() -> None:
    """The researcher-facing example must not depend on the toolkit it will be adapted to.

    The docstring is allowed to name `tailrisk` in prose; the *imports* are what decide whether
    the file is standalone, so they are checked from the syntax tree rather than from the text.
    """
    tree = ast.parse(STANDALONE_MODEL.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= {"__future__", "numpy", "scipy"}, (
        f"{STANDALONE_MODEL.name} imports {sorted(modules)}; keep it standalone"
    )
    assert "tailrisk" not in modules


def test_standalone_model_exposes_the_documented_interface() -> None:
    """`forecast(history, alphas, *, decay, warmup)` with the documented defaults."""
    spec = importlib.util.spec_from_file_location("tailrisk_example_ewma", STANDALONE_MODEL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    signature = inspect.signature(module.forecast)
    assert list(signature.parameters) == ["history", "alphas", "decay", "warmup"]
    assert signature.parameters["decay"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["warmup"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["decay"].default == 0.94
    assert signature.parameters["warmup"].default == 63

    var, es, sigma = module.forecast([0.01] * 63 + [0.02, 0.02], [0.025, 0.01])
    assert var.shape == es.shape == (2,)
    assert isinstance(sigma, float)
    assert np.all(np.isfinite(var)) and np.all(np.isfinite(es))
    assert np.all(es <= var) and np.all(var < 0.0)


def test_standalone_model_accepts_exactly_warmup_observations() -> None:
    """`warmup` observations forecast from the initial estimate; one fewer is refused."""
    spec = importlib.util.spec_from_file_location("tailrisk_example_ewma_floor", STANDALONE_MODEL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _, _, sigma_at_floor = module.forecast([0.01] * 63, [0.025])
    assert sigma_at_floor**2 == pytest.approx(1.0e-4, rel=1e-15)

    _, _, sigma_one_more = module.forecast([0.01] * 63 + [0.02], [0.025])
    assert sigma_one_more**2 == pytest.approx(1.18e-4, rel=1e-15)

    with pytest.raises(ValueError):
        module.forecast([0.01] * 62, [0.025])
