"""Build the profile-derived journey.

One job: `build_profile_journey` reads `tenant_profile/` into a
`ProfileJourney`. No LLM call, no disk write, no share round trip beyond the
profile read the pipeline already does.

V9 removed the second job. `build_journey_index` / `score_questions` used to
embed the leaves and rank them per question so the prompt could carry a top-4
instead of the whole journey. That went away with the `sentence-transformers`
dependency: the cut list was repeated for every question in the batched prompt
while the full list is sent once, so retrieval was costing input tokens rather
than saving them, and a correct leaf ranked below the cut could not be recovered
by the model. Selection is now the LLM's alone, over the complete catalog —
see `ProfileJourney.catalog`.
"""

from __future__ import annotations

import logging

from models.journey import (
    JourneyLeaf,
    JourneyType,
    ProfileJourney,
    compute_source_hash,
    normalize_name,
    slugify,
)
from models.tenant_profile import TenantProfile

logger = logging.getLogger(__name__)

CX_SOURCE_FIELD = "cx.journeys[].stages[]"
EX_SOURCE_FIELD = "ex.lifecycle_analysis.stages[]"

_EX_JOURNEY_NAME = "Employee Lifecycle"


# ---------- Build from profile ----------


def build_profile_journey(
    tenant_id: int,
    profile: TenantProfile | None,
    journey_type: JourneyType,
) -> ProfileJourney | None:
    """Read one journey type out of the tenant profile.

    Returns None when the profile is absent or carries no usable stages — the
    caller then leaves the journey dimensions unassigned rather than falling
    back to a generic template. A fabricated stage that reads as grounded is
    worse for a dashboard than an explicit gap.
    """
    if profile is None:
        return None

    leaves = (
        _cx_leaves(profile) if journey_type == "CX" else _ex_leaves(profile)
    )
    if not leaves:
        return None

    journey = ProfileJourney(
        tenant_id=tenant_id,
        journey_type=journey_type,
        journey_name=_journey_name(profile, journey_type),
        leaves=leaves,
        source_field=CX_SOURCE_FIELD if journey_type == "CX" else EX_SOURCE_FIELD,
        source_hash=compute_source_hash(leaves),
    )
    logger.info(
        "profile_journey_built",
        extra={
            "tenant_id": tenant_id,
            "journey_type": journey_type,
            "leaves": len(leaves),
            "stages": len(journey.stage_values),
            "has_sub_stages": journey.has_sub_stages,
        },
    )
    return journey


def _cx_leaves(profile: TenantProfile) -> list[JourneyLeaf]:
    """CX is two-level: journey -> stage. The journey name becomes
    `journey_stage`, the stage name becomes `sub_stage_name`.

    A journey carrying no `stages[]` still yields one leaf so its questions can
    be placed at the parent level; that leaf simply has no sub-stage.
    """
    if not profile.has_cx:
        return []

    out: list[JourneyLeaf] = []
    seen: set[str] = set()
    for journey in profile.cx_journeys:
        if not isinstance(journey, dict):
            continue
        journey_name = str(journey.get("journey_name") or "").strip()
        if not journey_name:
            continue
        stages = journey.get("stages")
        stages = [s for s in stages if isinstance(s, dict)] if isinstance(stages, list) else []

        if not stages:
            _append(out, seen, JourneyLeaf(
                leaf_id=slugify(journey_name) or "stage",
                stage_value=journey_name,
                sub_stage_value=None,
                description=str(journey.get("description") or "").strip(),
            ))
            continue

        for stage in stages:
            stage_name = str(stage.get("stage_name") or stage.get("name") or "").strip()
            if not stage_name:
                continue
            _append(out, seen, JourneyLeaf(
                leaf_id=f"{slugify(journey_name)}--{slugify(stage_name)}",
                stage_value=journey_name,
                sub_stage_value=stage_name,
                description=str(stage.get("description") or "").strip(),
                goal=str(stage.get("customer_goal") or "").strip(),
            ))
    return out


def _ex_leaves(profile: TenantProfile) -> list[JourneyLeaf]:
    """EX is one-level: the lifecycle stage becomes `journey_stage` and there is
    no sub-stage. The agent schema has no second level to read, so we assign
    none — the alternative is letting the model invent one."""
    if not profile.has_ex:
        return []

    out: list[JourneyLeaf] = []
    seen: set[str] = set()
    for stage in profile.ex_lifecycle_stages:
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("stage_name") or stage.get("name") or "").strip()
        if not stage_name:
            continue
        _append(out, seen, JourneyLeaf(
            leaf_id=slugify(stage_name) or "stage",
            stage_value=stage_name,
            sub_stage_value=None,
            description=str(stage.get("description") or "").strip(),
            goal=str(stage.get("employee_goal") or stage.get("customer_goal") or "").strip(),
        ))
    return out


def _append(out: list[JourneyLeaf], seen: set[str], leaf: JourneyLeaf) -> None:
    """De-dup on the (stage, sub_stage) pair, not on the leaf name alone.

    Deliberately narrower than the canon's dedup, which collapsed same-named
    stages across different journeys and so lost the parent level. Two journeys
    may legitimately both contain a "Feedback" stage; those are different
    moments and must stay separate.
    """
    key = f"{normalize_name(leaf.stage_value)}|{normalize_name(leaf.sub_stage_value or '')}"
    if key in seen:
        return
    seen.add(key)
    out.append(leaf)


def _journey_name(profile: TenantProfile, journey_type: JourneyType) -> str:
    if journey_type == "EX":
        return _EX_JOURNEY_NAME
    industry = (profile.industry_taxonomy_vertical or profile.industry_vertical or "").strip()
    return f"{industry} Customer Journey".strip() if industry else "Customer Journey"
