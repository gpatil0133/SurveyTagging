"""Taxonomy registry — loads and validates tag dimensions from config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TaxonomyDimension(BaseModel):
    """A single tag dimension with its allowed values."""

    name: str
    level: str  # "project" or "question"
    description: str = ""
    allowed_values: list[str] = Field(default_factory=list)
    multi_label: bool = False
    user_defined: bool = False
    canonical_values: list[str] = Field(default_factory=list)
    """Preferred values for user_defined dims — shown to LLM as strong suggestions
    but not strictly enforced. Used for dashboard_routing/dashboard_placement."""

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

        for level in ("project_level", "question_level"):
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
    def project_dimensions(self) -> list[TaxonomyDimension]:
        return [d for d in self._dimensions.values() if d.level == "project"]

    @property
    def question_dimensions(self) -> list[TaxonomyDimension]:
        return [d for d in self._dimensions.values() if d.level == "question"]

    @property
    def all_dimensions(self) -> dict[str, TaxonomyDimension]:
        return dict(self._dimensions)
