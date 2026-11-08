"""Explicit model-factory resolution for built-in and user-supplied adapters.

The built-in models are constructs of this package, so their factories are ordinary
functions in :mod:`tailrisk.models.adapters`. A *custom* model is different: the toolkit
does not know what it needs, so the configuration names one importable
``module:function`` and this module resolves it, records where it came from, and hands
back the adapter.

What is deliberately absent
---------------------------
There is no filesystem discovery, no entry-point registry, no ``eval``, no package
installation and no network access. A factory is reachable only because a configuration
names it explicitly, and only because the module is already importable in the process.
That keeps "which code ran" answerable from the run's own configuration rather than from
whatever happened to be installed.

Importing a factory executes trusted Python at import time. This is a local research
toolkit, not a sandbox: the provenance block records which file was imported and its hash,
and that is the extent of the claim. Hashing one module says nothing about the rest of its
dependency tree, and nothing here verifies that a checkpoint was not pretrained elsewhere.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adapters import BUILTIN_MODEL_IDS, create_model
from .protocol import CONTRACT_VERSION, ContractError

#: Config keys a ``custom_models`` entry may carry.
CUSTOM_ENTRY_KEYS = ("factory", "settings", "seed", "dependencies")

#: ``module:function`` — one separator, no empty side, no whitespace.
_FACTORY_PATTERN = re.compile(r"^(?P<module>[A-Za-z_][\w.]*):(?P<function>[A-Za-z_]\w*)$")


@dataclass(frozen=True)
class CustomFactory:
    """One resolved custom factory, with the configuration that named it."""

    model_id: str
    factory: str
    module_name: str
    function_name: str
    settings: Mapping[str, Any]
    seed: int | None
    dependencies: tuple[str, ...]

    def build(self) -> Any:
        """Call the factory and return the adapter it produced.

        The call is keyword-only ``settings=`` and ``seed=``, which is the whole of the
        public factory contract: a custom adapter receives no alpha grid, no runner object
        and no data. An alpha-dependent model declares the levels it was trained at inside
        its own settings.
        """
        function = _import_factory(self.module_name, self.function_name)
        try:
            adapter = function(settings=dict(self.settings), seed=self.seed)
        except ContractError:
            raise
        except Exception as exc:  # noqa: BLE001 - the factory is user code
            raise ContractError(
                f"custom model {self.model_id!r}: factory {self.factory!r} raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return _validate_adapter(self.model_id, self.factory, adapter)


def _import_factory(module_name: str, function_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ContractError(
            f"custom factory module {module_name!r} is not importable ({exc}); "
            "install its dependencies and run from a directory where it resolves"
        ) from exc
    function = getattr(module, function_name, None)
    if function is None:
        raise ContractError(
            f"custom factory {module_name}:{function_name} does not exist; the module "
            f"defines {sorted(name for name in vars(module) if not name.startswith('_'))[:8]}"
        )
    if not callable(function):
        raise ContractError(f"custom factory {module_name}:{function_name} is not callable")
    return function


def _validate_adapter(model_id: str, factory: str, adapter: Any) -> Any:
    """The returned object must announce itself as the model the config selected."""
    spec = getattr(adapter, "spec", None)
    if spec is None:
        raise ContractError(
            f"custom model {model_id!r}: factory {factory!r} returned "
            f"{type(adapter).__name__} without a .spec ModelSpec"
        )
    if spec.contract_version != CONTRACT_VERSION:
        raise ContractError(
            f"custom model {model_id!r}: adapter declares contract_version "
            f"{spec.contract_version}, this toolkit supports {CONTRACT_VERSION}"
        )
    if spec.model_id != model_id:
        raise ContractError(
            f"custom model {model_id!r}: factory {factory!r} returned an adapter with "
            f"model_id {spec.model_id!r}; the two must agree so persisted rows and the "
            "configuration cannot disagree about which model produced them"
        )
    for method in ("fit", "predict_one", "summary"):
        if not callable(getattr(adapter, method, None)):
            raise ContractError(f"custom model {model_id!r}: adapter has no callable {method}()")
    if getattr(spec, "replay_mode", "none") == "checkpoint":
        # A checkpoint nobody can load is not reusable state: the run would ship files that
        # only the fitting process could interpret. Requiring the loader here means the
        # promise is checkable at registration time rather than at verification time.
        module_name = factory.split(":", 1)[0]
        module = sys.modules.get(module_name)
        if module is None or not callable(getattr(module, "load_checkpoint", None)):
            raise ContractError(
                f"custom model {model_id!r}: replay_mode='checkpoint' requires a module-level "
                f"load_checkpoint(directory) in {module_name!r}, so saved state can be "
                "replayed without fitting"
            )
    return adapter


def parse_factory_reference(model_id: str, entry: Mapping[str, Any]) -> CustomFactory:
    """Validate one ``custom_models`` entry without importing anything yet."""
    if not isinstance(entry, Mapping):
        raise ContractError(f"custom model {model_id!r}: configuration must be a JSON object")
    unknown = sorted(set(entry) - set(CUSTOM_ENTRY_KEYS))
    if unknown:
        raise ContractError(
            f"custom model {model_id!r}: unknown key(s) {unknown}; "
            f"allowed: {list(CUSTOM_ENTRY_KEYS)}"
        )
    if model_id in BUILTIN_MODEL_IDS:
        raise ContractError(
            f"custom model {model_id!r} collides with a built-in model; built-in ids are "
            f"{list(BUILTIN_MODEL_IDS)} and cannot be redefined"
        )

    factory = entry.get("factory")
    if not isinstance(factory, str):
        raise ContractError(f"custom model {model_id!r}: 'factory' must be an 'module:function' string")
    match = _FACTORY_PATTERN.match(factory.strip())
    if match is None:
        raise ContractError(
            f"custom model {model_id!r}: factory {factory!r} is not of the form "
            "'module:function' (for example 'examples.custom_model.model:create_model')"
        )

    settings = entry.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ContractError(f"custom model {model_id!r}: 'settings' must be a JSON object")

    seed = entry.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ContractError(f"custom model {model_id!r}: 'seed' must be an integer or absent")
        if seed < 0:
            raise ContractError(f"custom model {model_id!r}: 'seed' must be non-negative")

    dependencies = entry.get("dependencies", ())
    if isinstance(dependencies, str) or not isinstance(dependencies, (list, tuple)):
        raise ContractError(
            f"custom model {model_id!r}: 'dependencies' must be a list of package names"
        )
    if any(not isinstance(item, str) or not item.strip() for item in dependencies):
        raise ContractError(
            f"custom model {model_id!r}: 'dependencies' entries must be nonempty strings"
        )

    return CustomFactory(
        model_id=str(model_id),
        factory=factory.strip(),
        module_name=match.group("module"),
        function_name=match.group("function"),
        settings=dict(settings),
        seed=None if seed is None else int(seed),
        dependencies=tuple(dependencies),
    )


def resolve_custom_models(
    custom_models: Any, *, selected: Any
) -> dict[str, CustomFactory]:
    """Resolve the run's custom model set, or raise before any data is loaded.

    Every entry is checked against the selection, so a configuration cannot carry a model
    it never runs (an unused entry is a stale claim about what the experiment contained)
    and cannot run a model it never configured.
    """
    if custom_models is None:
        custom_models = {}
    if not isinstance(custom_models, Mapping):
        raise ContractError("'custom_models' must be a JSON object mapping model id to configuration")
    if not isinstance(selected, (list, tuple)) or not all(isinstance(item, str) for item in selected):
        raise ContractError("'models' must be a list of model id strings")

    duplicates = sorted({name for name in selected if list(selected).count(name) > 1})
    if duplicates:
        raise ContractError(f"'models' repeats {duplicates}; each model runs once")

    resolved: dict[str, CustomFactory] = {}
    for model_id, entry in custom_models.items():
        if model_id not in selected:
            raise ContractError(
                f"custom model {model_id!r} is configured but not selected in 'models' "
                f"{list(selected)}; remove the entry or run it"
            )
        resolved[str(model_id)] = parse_factory_reference(str(model_id), entry)

    undeclared = [name for name in selected if name not in BUILTIN_MODEL_IDS and name not in resolved]
    if undeclared:
        raise ContractError(
            f"'models' names {undeclared}, which is neither a built-in model "
            f"{list(BUILTIN_MODEL_IDS)} nor declared under 'custom_models'"
        )
    return resolved


def create_registered_model(
    model_id: str,
    *,
    settings: Mapping[str, Any],
    alphas: Any | None = None,
    seed: int | None = None,
    custom: Mapping[str, CustomFactory] | None = None,
) -> Any:
    """Build one selected model, built-in or custom.

    A built-in id keeps the built-in factory and its settings/seed validation. A custom id
    must have been resolved already; the registry never imports a module that the run's
    configuration did not name.
    """
    if model_id in BUILTIN_MODEL_IDS:
        return create_model(model_id, settings=settings, alphas=alphas, seed=seed)
    registry = custom or {}
    factory = registry.get(model_id)
    if factory is None:
        raise ContractError(
            f"model {model_id!r} is not a built-in id {list(BUILTIN_MODEL_IDS)} and has no "
            "resolved custom factory"
        )
    if settings:
        raise ContractError(
            f"custom model {model_id!r}: settings come from its 'custom_models' entry, "
            "not from the built-in settings resolver"
        )
    return factory.build()


def load_custom_checkpoint(factory: CustomFactory, directory: str | Path) -> Any:
    """Rebuild a custom adapter from the checkpoint its factory module wrote.

    This is the loader half of the checkpoint capability: the *module* provides
    ``load_checkpoint(directory)``, not the factory, because loading a saved state is a
    different operation from constructing a fresh adapter. The loaded object is validated
    against the same spec rules as a freshly built one, so a checkpoint cannot be used to
    smuggle in an adapter that declares a different id or contract version.

    Nothing here calls ``fit``; that is the point of replay.
    """
    module = importlib.import_module(factory.module_name)
    loader = getattr(module, "load_checkpoint", None)
    if not callable(loader):
        raise ContractError(
            f"custom model {factory.model_id!r}: module {factory.module_name!r} has no "
            "callable load_checkpoint(directory), so its checkpoint cannot be replayed"
        )
    adapter = loader(Path(directory))
    return _validate_adapter(factory.model_id, factory.factory, adapter)


def factory_provenance(factory: CustomFactory) -> dict[str, Any]:
    """What can be said about where a custom factory came from.

    A missing file path, hash or distribution version is recorded as ``None`` with a note,
    never filled in with a guess: "the module is the one we imported" and "the module's
    source is exactly this file" are different claims, and only the first is always true.
    """
    module = sys.modules.get(factory.module_name)
    module_file = getattr(module, "__file__", None)
    source_hash = None
    hash_note = "module has no source file (namespace or built-in module)"
    if module_file:
        try:
            source_hash = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
            hash_note = "sha256 of the module file this process imported"
        except OSError as exc:
            hash_note = f"module file is unreadable: {exc}"

    distribution = factory.module_name.split(".", 1)[0]
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
        distribution_note = (
            f"top-level name {distribution!r} is not an installed distribution, so no "
            "package version exists; the source hash identifies this file only"
        )
    else:
        distribution_note = f"version of the installed distribution named {distribution!r}"

    return {
        "model_id": factory.model_id,
        "factory": factory.factory,
        "module": factory.module_name,
        "function": factory.function_name,
        "module_file": module_file,
        "module_sha256": source_hash,
        "module_hash_note": hash_note,
        "distribution": distribution if version is not None else None,
        "distribution_version": version,
        "distribution_note": distribution_note,
        "declared_dependencies": list(factory.dependencies),
        "dependencies_note": "author declaration; not installed, resolved or verified by this run",
        "trust_note": (
            "importing a factory executes trusted local Python; hashing one module is not "
            "a statement about its dependencies, its imports at call time, or any "
            "pretraining behind its checkpoint"
        ),
    }


__all__ = [
    "CUSTOM_ENTRY_KEYS",
    "CustomFactory",
    "parse_factory_reference",
    "resolve_custom_models",
    "create_registered_model",
    "load_custom_checkpoint",
    "factory_provenance",
]
