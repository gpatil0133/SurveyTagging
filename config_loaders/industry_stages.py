"""Industry-specific journey stage registry loaded from journey_stages.yaml."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


_HARDCODED_DEFAULT_STAGES = [
    "Awareness", "Consideration", "Onboarding",
    "Active", "Retention", "Advocacy", "Churn",
]
_HARDCODED_GENERIC_SURVEY = ["Pre-Survey", "During", "Post-Survey"]


class IndustryStagesRegistry:
    """Loads industry-specific journey stages from journey_stages.yaml.

    Used by JourneyStageTagger (deterministic priors + LLM prompt construction)
    and by ResponseParser (fuzzy matching LLM output to industry's stages).
    """

    def __init__(self) -> None:
        self._stages: dict[str, list[str]] = {}
        self._stage_keywords: dict[str, list[str]] = {}
        self._role_to_stage: dict[str, dict[str, str]] = {}
        self._cx_default: dict = {}
        self._ex_default: dict = {}
        self._canonical_reduction: dict[str, dict[str, str]] = {}

    @classmethod
    def from_yaml(cls, path: Path) -> IndustryStagesRegistry:
        """Load registry from YAML. Falls back to hardcoded defaults if file missing."""
        registry = cls()
        if not path.exists():
            logger.warning("journey_stages_file_missing",
                           extra={"path": str(path), "fallback": "hardcoded_default"})
            registry._install_hardcoded_fallback()
            return registry

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            registry._stages = data.get("stages", {})
            registry._stage_keywords = data.get("stage_keywords", {})
            registry._role_to_stage = data.get("role_to_stage", {})
            registry._cx_default = data.get("_cx_default", {})
            registry._ex_default = data.get("_ex_default", {})
            registry._canonical_reduction = data.get("canonical_reduction", {})
        except (yaml.YAMLError, OSError) as e:
            logger.error("journey_stages_load_failed",
                         extra={"path": str(path), "error": str(e)})
            registry._install_hardcoded_fallback()

        # Ensure safety floor: always have _default and _generic_survey
        if "_default" not in registry._stages:
            registry._stages["_default"] = _HARDCODED_DEFAULT_STAGES
        if "_generic_survey" not in registry._stages:
            registry._stages["_generic_survey"] = _HARDCODED_GENERIC_SURVEY

        logger.info("industry_stages_loaded",
                    extra={"industries": len(registry._stages)})
        return registry

    def _install_hardcoded_fallback(self) -> None:
        self._stages = {
            "_default": _HARDCODED_DEFAULT_STAGES,
            "_generic_survey": _HARDCODED_GENERIC_SURVEY,
        }
        self._stage_keywords = {}
        self._role_to_stage = {}

    def get_stages(self, industry_vertical: str | None, project_type: str | None = None) -> list[str]:
        """Return the ordered stage list for an industry.

        Args:
            industry_vertical: Project industry tag value.
            project_type: Project type ("CX", "EX", "Survey", "Assessment").

        Returns:
            Ordered list of stage names (first = earliest in lifecycle).
        """
        if project_type in ("Survey", "Assessment"):
            return list(self._stages.get("_generic_survey", _HARDCODED_GENERIC_SURVEY))
        if industry_vertical and industry_vertical in self._stages:
            return list(self._stages[industry_vertical])
        return list(self._stages.get("_default", _HARDCODED_DEFAULT_STAGES))

    def get_stage_keywords(self) -> dict[str, list[str]]:
        """Return the keyword -> stage_role map for heuristic priors."""
        return dict(self._stage_keywords)

    def role_to_stage(self, industry_vertical: str | None, role: str,
                      project_type: str | None = None) -> str | None:
        """Map a stage_role (e.g., 'onboarding') to the industry's actual stage name.

        Returns None if role not mapped for this industry.
        """
        if project_type in ("Survey", "Assessment"):
            key = "_generic_survey"
        elif industry_vertical and industry_vertical in self._role_to_stage:
            key = industry_vertical
        else:
            key = "_default"

        mapping = self._role_to_stage.get(key, {})
        return mapping.get(role)

    def all_stage_names(self) -> set[str]:
        """All known stage names across industries (for fuzzy validation)."""
        names: set[str] = set()
        for stages in self._stages.values():
            names.update(stages)
        return names

    def get_default_canonical(self, journey_type: str) -> dict:
        """Return the default CX or EX canonical journey structure.

        Shape: {"journey_name": str, "stages": [{"name": str, "description": str}, ...]}.
        Returns an empty dict if the journey_type is unknown or the YAML omits the block.
        """
        key = (journey_type or "").strip().lower()
        if key == "cx":
            return dict(self._cx_default)
        if key == "ex":
            return dict(self._ex_default)
        return {}

    def reduce_to_canonical(self, journey_type: str, industry_stage: str | None) -> str | None:
        """Reduce an industry-specific journey_stage value onto the CX/EX canonical name.

        Returns None when unmapped — callers should decide a fallback stage.
        """
        if not industry_stage:
            return None
        key = (journey_type or "").strip().lower()
        mapping = self._canonical_reduction.get(key, {})
        return mapping.get(industry_stage)
