"""Tagger registry: auto-discovery, dependency resolution, and execution ordering."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections import defaultdict

from taggers.base import BaseTagger

logger = logging.getLogger(__name__)


class TaggerRegistry:
    """Auto-discovers and manages all tagger plugins."""

    def __init__(self) -> None:
        self._taggers: dict[str, BaseTagger] = {}
        self._order: list[list[BaseTagger]] | None = None

    def register(self, tagger: BaseTagger) -> None:
        """Register a single tagger instance."""
        if tagger.name in self._taggers:
            raise ValueError(f"Duplicate tagger name: {tagger.name}")
        self._taggers[tagger.name] = tagger
        self._order = None      # a new tagger invalidates the resolved order
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
        """Execution order: stage number first, `depends_on` within a stage.

        Returns one list per stage, in the order the stages run. Taggers inside a
        stage are ordered so that a dependency always precedes its dependent.

        Both rules are needed. Stage number is the coarse contract every tagger
        declares, but four taggers depend on another in the SAME stage
        (`segment_dimensions` -> `is_segmentable`, `sub_stage_name` ->
        `journey_stage`, `display_role` -> `dashboard_placement`, and
        `project.audience` -> `project.project_type`). Grouping by stage alone left
        their order to `pkgutil`, i.e. to alphabetical filename — three of the four
        were correct only because the dependency's file happened to sort first, and
        renaming a file would have silently broken the tagger that reads its output
        (an unmet dependency reads as `None`, not as an error).

        A dependency in a LATER stage cannot be honoured by reordering and is
        reported rather than silently tolerated; so is a cycle, which keeps the
        remaining taggers running in declaration order instead of not at all.

        Cached: the result is a pure function of the registered taggers, and this
        is called once per survey.
        """
        if self._order is not None:
            return self._order

        for tagger in self._taggers.values():
            for dep in tagger.depends_on:
                other = self._taggers.get(dep)
                if other is None:
                    logger.warning("missing_dependency",
                                   extra={"tagger": tagger.name, "depends_on": dep})
                elif other.stage > tagger.stage:
                    logger.warning("dependency_runs_later",
                                   extra={"tagger": tagger.name, "depends_on": dep,
                                          "tagger_stage": tagger.stage,
                                          "dependency_stage": other.stage})

        stage_groups: dict[int, list[BaseTagger]] = defaultdict(list)
        for tagger in self._taggers.values():
            stage_groups[tagger.stage].append(tagger)

        self._order = [
            self._sort_within_stage(stage_groups[stage_num], stage_num)
            for stage_num in sorted(stage_groups)
        ]
        return self._order

    def _sort_within_stage(
        self, taggers: list[BaseTagger], stage_num: int
    ) -> list[BaseTagger]:
        """Kahn topological sort over same-stage `depends_on` edges.

        Ties keep declaration order, so the output is stable across runs. On a
        cycle the unresolved remainder is appended in declaration order and logged:
        a wrong order degrades some evidence, refusing to run degrades everything.
        """
        names = {t.name for t in taggers}
        pending = list(taggers)
        emitted: set[str] = set()
        ordered: list[BaseTagger] = []

        while pending:
            ready = [t for t in pending
                     if all(d in emitted for d in t.depends_on if d in names)]
            if not ready:
                logger.error(
                    "tagger_dependency_cycle",
                    extra={"stage": stage_num,
                           "taggers": ", ".join(t.name for t in pending)},
                )
                ordered.extend(pending)
                break
            for t in ready:
                ordered.append(t)
                emitted.add(t.name)
            pending = [t for t in pending if t.name not in emitted]

        return ordered

    @property
    def all_taggers(self) -> dict[str, BaseTagger]:
        return dict(self._taggers)

    @property
    def project_taggers(self) -> list[BaseTagger]:
        return [t for t in self._taggers.values() if t.level == "project"]

    @property
    def question_taggers(self) -> list[BaseTagger]:
        return [t for t in self._taggers.values() if t.level == "question"]
