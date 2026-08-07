"""Shared helpers for projection modules.

Extracted so [projections/journey.py](../projections/journey.py) and
[projections/custom_journey.py](../projections/custom_journey.py) can reuse
the same tag-extraction logic and stage synonym map.
"""

from __future__ import annotations

from typing import Any


# Common synonym normalization for stage-like strings emitted by the LLM.
# Keeps minor LLM variations consistent before further canonicalization.
STAGE_SYNONYMS: dict[str, str] = {
    "onboard": "Onboarding",
    "on-boarding": "Onboarding",
    "getting started": "Onboarding",
    "sign up": "Acquisition",
    "signup": "Acquisition",
    "post-onboarding": "Adoption",
    "in-use": "Active",
    "active use": "Active",
    "advocate": "Advocacy",
    "promote": "Advocacy",
    "exit": "Churn",
    "cancel": "Churn",
    "cancellation": "Churn",
}


def normalize_stage(raw: str) -> tuple[str, bool]:
    """Return (normalized, was_changed). Empty input passes through."""
    if not raw:
        return raw, False
    lower = raw.lower().strip()
    if lower in STAGE_SYNONYMS:
        return STAGE_SYNONYMS[lower], True
    if raw != raw.strip() or raw.islower():
        return raw.strip().title(), True
    return raw, False


def get_tag_value(tags: dict, dim: str) -> Any:
    """Extract .value from a tag entry. Handles both dict-form and scalar forms."""
    t = tags.get(dim)
    if t is None:
        return None
    if isinstance(t, dict):
        return t.get("value")
    return t


def get_tag_field(tags: dict, dim: str, field: str) -> Any:
    """Extract an arbitrary field from a tag entry (status, confidence,
    coverage_metadata, etc.). Returns None for scalar-form tags or missing fields."""
    t = tags.get(dim)
    if isinstance(t, dict):
        return t.get(field)
    return None


def get_journey_confidence(tags: dict) -> str | None:
    """Pull the LLM's stated journey confidence from the journey_stage tag's
    coverage_metadata, with sensible defaults for legacy/missing entries."""
    cov = get_tag_field(tags, "journey_stage", "coverage_metadata")
    if isinstance(cov, dict):
        c = cov.get("confidence")
        if isinstance(c, str) and c:
            return c.lower()
    return None


def get_journey_status(tags: dict) -> str:
    """Return "assigned", "low_confidence_assigned", "skipped", "failed", or
    "missing" (legacy/no-tag) for the journey_stage tag."""
    s = get_tag_field(tags, "journey_stage", "status")
    if isinstance(s, str) and s:
        return s
    # No status field → infer from value presence (legacy v4 outputs)
    return "assigned" if get_tag_value(tags, "journey_stage") else "missing"
