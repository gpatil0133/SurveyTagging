"""Tenant-canon Pydantic models.

A `TenantCanon` is the single source of truth for journey stages of one tenant
+ journey type (CX or EX). It is built ONCE per tenant from the Parallel.ai
`TenantProfile` artifacts (with industry-template fallback) and used uniformly
from per-question tagging through final journey assembly. All downstream
journey/dashboard code reads from this canon — there is no parallel YAML
namespace at question-tag time anymore.

Sources (recorded on the artifact for transparency):
- "agent_canon"       — agent confidence High and stages rich; agent stages preserved verbatim
- "agent_blended"     — agent has signal but sparse / Medium confidence; LLM merges with template
- "industry_template" — no usable agent data; uses journey_stages.yaml templates
- "legacy_lifted"     — synthesized in-memory from a pre-v5 journey_stages_*.json file (transition)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CanonSource = Literal["agent_canon", "agent_blended", "industry_template", "legacy_lifted"]
JourneyType = Literal["CX", "EX"]


class CanonStage(BaseModel):
    """One stage in the tenant canon.

    `name` is the display name (always used as the namespace key downstream).
    `canon_id` is a stable slug used as a dict key when the display name might
    contain whitespace or punctuation; never rendered to users.
    """

    canon_id: str
    name: str
    description: str = ""
    customer_goal: str = ""
    synonyms: list[str] = Field(default_factory=list)
    source_journeys: list[str] = Field(default_factory=list)
    source_stage_names: list[str] = Field(default_factory=list)
    industry_template_match: str | None = None


class TenantCanon(BaseModel):
    """Per-tenant, per-journey-type canonical stage list."""

    schema_version: str = "1.0"
    tenant_id: int
    journey_type: JourneyType
    journey_name: str
    industry: str = ""
    source: CanonSource
    # `locked` means "human-approved — never auto-rebuild". It must default to
    # False: a freshly built canon is NOT approved, and defaulting it to True
    # froze every tenant's first (often profile-less) build forever. Auto-rebuild
    # is instead gated on `input_hash` (see get_or_build_tenant_canon_async).
    # Only lift_legacy_to_canon and an explicit approve action set this True.
    locked: bool = False
    derived_at: str
    confidence: str = ""
    stages: list[CanonStage] = Field(default_factory=list)
    input_hash: str = ""

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def find_stage(self, name: str) -> CanonStage | None:
        """Case-insensitive exact lookup by display name."""
        if not name:
            return None
        target = name.strip().lower()
        for s in self.stages:
            if s.name.strip().lower() == target:
                return s
        return None
