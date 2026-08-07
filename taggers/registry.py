"""Tagger registry: auto-discovery, dependency resolution, and execution ordering."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections import defaultdict
from typing import Type

from taggers.base import BaseTagger

logger = logging.getLogger(__name__)


class TaggerRegistry:
    """Auto-discovers and manages all tagger plugins."""

    def __init__(self) -> None:
        self._taggers: dict[str, BaseTagger] = {}

    def register(self, tagger: BaseTagger) -> None:
        """Register a single tagger instance."""
        if tagger.name in self._taggers:
            raise ValueError(f"Duplicate tagger name: {tagger.name}")
        self._taggers[tagger.name] = tagger
        logger.debug("tagger_registered",
                     extra={"tagger_name": tagger.name, "stage": tagger.stage})

    def discover(self, package_name: str) -> None:
        """Auto-discover and register all taggers in a package.

        Scans the package for modules containing classes that inherit from BaseTagger.
        Each module should define a tagger class; the module-level `create_tagger()`
        function or a class with a no-arg constructor will be instantiated.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.error("tagger_package_import_failed", extra={"package": package_name, "error": str(e)})
            return

        if not hasattr(package, "__path__"):
            return

        for _importer, module_name, _is_pkg in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("_"):
                continue
            full_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_name)
                # Look for a create_tagger() factory function
                if hasattr(module, "create_tagger"):
                    tagger = module.create_tagger()
                    if isinstance(tagger, BaseTagger):
                        self.register(tagger)
                    elif isinstance(tagger, list):
                        for t in tagger:
                            if isinstance(t, BaseTagger):
                                self.register(t)
            except Exception as e:
                logger.warning("tagger_module_load_failed",
                               extra={"module_name": full_name, "error": str(e)})

    def get_tagger(self, name: str) -> BaseTagger | None:
        return self._taggers.get(name)

    def resolve_execution_order(self) -> list[list[BaseTagger]]:
        """Topological sort of taggers by stage + dependencies.

        Returns a list of stages, where each stage is a list of taggers
        that can run in parallel (no inter-dependencies within a stage).
        """
        # Validate dependencies
        for tagger in self._taggers.values():
            for dep in tagger.depends_on:
                if dep not in self._taggers:
                    logger.warning("missing_dependency",
                                   extra={"tagger": tagger.name, "depends_on": dep})

        # Group by stage
        stage_groups: dict[int, list[BaseTagger]] = defaultdict(list)
        for tagger in self._taggers.values():
            stage_groups[tagger.stage].append(tagger)

        # Return sorted by stage number
        result = []
        for stage_num in sorted(stage_groups.keys()):
            result.append(stage_groups[stage_num])

        return result

    @property
    def all_taggers(self) -> dict[str, BaseTagger]:
        return dict(self._taggers)

    @property
    def project_taggers(self) -> list[BaseTagger]:
        return [t for t in self._taggers.values() if t.level == "project"]

    @property
    def question_taggers(self) -> list[BaseTagger]:
        return [t for t in self._taggers.values() if t.level == "question"]
