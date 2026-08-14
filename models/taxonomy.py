"""Taxonomy registry — loads and validates tag dimensions from config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TaxonomyDimension(BaseModel):
    """A single tag dimension with its allowed values."""

    name: str
    level: str  # "tenant", "project" or "question"
    description: str = ""
    allowed_values: list[str] = Field(default_factory=list)
    multi_label: bool = False
    user_defined: bool = False
    canonical_values: list[str] = Field(default_factory=list)
    """Preferred values for user_defined dims — shown to LLM as strong suggestions
    but not strictly enforced. Used for dashboard_routing/dashboard_placement."""

    # --- explanation layer (V7) ---------------------------------------------
    # Documentation only: nothing validates against these and no tagger reads
    # them. They exist so a reader meeting a tag for the first time can answer
    # "what is this?" and "where did the value come from?" without opening the
    # tagger. Served by GET /api/taxonomy and rendered in the UI Taxonomy tab.
    # Default to "" rather than being required, so a dimension added without
    # them still loads (it just shows blank columns).
    explanation: str = ""
    """What the dimension is, in plain language, and what consumes it.

    NOT documentation-only: `prompt_builder._dimension_guide` renders this into
    the project prompt's cached preamble, so editing it changes what the model
    is told and invalidates that prompt's cached responses. Keep it accurate and
    bump `project_tagging.yaml`'s version when it changes materially.
    """
    derivation: str = ""
    """Which inputs are read, the shape of the logic, the tagger module, and who
    gets the final say (rule vs LLM call)."""
    strategy: str = ""
    """One-word label for the derivation: deterministic | statistical | hybrid |
    llm-refined | llm-only | placeholder. See config/taxonomy.yaml's header."""

    def is_valid_value(self, value: str) -> bool:
        if self.user_defined:
            return True  # Free-text, anything goes
        return value in self.allowed_values

    def validate_tag(self, value: str | list[str] | None) -> list[str]:
        """Return list of invalid values (empty if all valid)."""
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        if self.user_defined:
            return []
        return [v for v in values if v not in self.allowed_values]


class TaxonomyRegistry:
    """Loads taxonomy from YAML and provides validation."""

    def __init__(self) -> None:
        self._dimensions: dict[str, TaxonomyDimension] = {}

    @classmethod
    def from_yaml(cls, path: Path) -> TaxonomyRegistry:
        registry = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Tenant dims are loaded too, so the catalog can describe all three
        # levels. They are never validated against (no LLM assigns them) and
        # `project_dimensions` / `question_dimensions` filter on level, so the
        # only behavioural change is that they now appear in `all_dimensions`.
        for level in ("tenant_level", "project_level", "question_level"):
            level_key = level.replace("_level", "")
            for dim_name, dim_config in data.get(level, {}).items():
                registry._dimensions[dim_name] = TaxonomyDimension(
                    name=dim_name,
                    level=level_key,
                    description=dim_config.get("description", ""),
                    allowed_values=dim_config.get("allowed_values", []),
                    multi_label=dim_config.get("multi_label", False),
                    user_defined=dim_config.get("user_defined", False),
                    canonical_values=dim_config.get("canonical_values", []),
                    explanation=dim_config.get("explanation", ""),
                    derivation=dim_config.get("derivation", ""),
                    strategy=dim_config.get("strategy", ""),
                )
        return registry

    def get_dimension(self, name: str) -> TaxonomyDimension | None:
        return self._dimensions.get(name)

    def validate_value(self, dimension: str, value: str | list[str] | None) -> list[str]:
        dim = self._dimensions.get(dimension)
        if dim is None:
            return [f"Unknown dimension: {dimension}"]
        return dim.validate_tag(value)

    def get_allowed_values(self, dimension: str) -> list[str]:
        dim = self._dimensions.get(dimension)
        return dim.allowed_values if dim else []

    @property
    def tenant_dimensions(self) -> list[TaxonomyDimension]:
        return [d for d in self._dimensions.values() if d.level == "tenant"]

    @property
    def project_dimensions(self) -> list[TaxonomyDimension]:
        return [d for d in self._dimensions.values() if d.level == "project"]

    @property
    def question_dimensions(self) -> list[TaxonomyDimension]:
        return [d for d in self._dimensions.values() if d.level == "question"]

    @property
    def all_dimensions(self) -> dict[str, TaxonomyDimension]:
        return dict(self._dimensions)
