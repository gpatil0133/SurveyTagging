"""Key CX touchpoints tagger.

Extracts a flat list of semantic touchpoint / journey-stage names from
tenant_profile.cx_journeys. The backend dashboard writer aligns widgets with
these touchpoints when deciding cross-tabs and per-stage layouts.

Output: list[str] of Title-Cased touchpoint names (no IDs, no codes).
"""

from __future__ import annotations

from typing import Any

from models import evidence as ev
from models.tags import TagResult
from models.tenant_profile import TenantProfile
from taggers.tenant.base import TenantTagger


def _clean(label: str) -> str:
    return " ".join(label.strip().split()).title()


def _collect_touchpoints(journeys: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}  # ordered de-dup
    for journey in journeys:
        # Per-journey list of stages; each stage may carry a name or
        # touchpoints[]. Agent schema is sparse — try a few shapes.
        stages = journey.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                name = stage.get("name") or stage.get("stage_name")
                if isinstance(name, str) and name.strip():
                    seen[_clean(name)] = None
                touchpoints = stage.get("touchpoints")
                if isinstance(touchpoints, list):
                    for tp in touchpoints:
                        if isinstance(tp, str) and tp.strip():
                            seen[_clean(tp)] = None
                        elif isinstance(tp, dict):
                            tp_name = tp.get("name") or tp.get("touchpoint")
                            if isinstance(tp_name, str) and tp_name.strip():
                                seen[_clean(tp_name)] = None
        # Some agents emit top-level touchpoints[] per journey instead of
        # nesting under stages
        top_touchpoints = journey.get("touchpoints")
        if isinstance(top_touchpoints, list):
            for tp in top_touchpoints:
                if isinstance(tp, str) and tp.strip():
                    seen[_clean(tp)] = None
                elif isinstance(tp, dict):
                    tp_name = tp.get("name") or tp.get("touchpoint")
                    if isinstance(tp_name, str) and tp_name.strip():
                        seen[_clean(tp_name)] = None
    return list(seen.keys())


class KeyCxTouchpointsTagger(TenantTagger):
    name = "tenant.key_cx_touchpoints"
    tag_dimension = "key_cx_touchpoints"
    source_type = "deterministic"

    def tag(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> TagResult:
        if tenant_profile is None or not tenant_profile.has_cx:
            return TagResult(
                value=[],
                source="deterministic",
                confidence=0.40,
                evidence=ev.fallback(
                    "tenant.cx_touchpoints.no_cx_profile",
                    "No CX intelligence artifact on disk for this tenant, so there were "
                    "no journeys to read touchpoints from. Empty list is 'unknown', not "
                    "'this tenant has no touchpoints'.",
                    inputs={"has_cx": False},
                ),
            )

        journeys = tenant_profile.cx_journeys
        touchpoints = _collect_touchpoints(journeys)
        if not touchpoints:
            return TagResult(
                value=[],
                source="deterministic",
                confidence=0.50,
                evidence=ev.profile(
                    "tenant.cx_touchpoints.none_extracted",
                    f"The CX profile has {len(journeys)} journey(s) but none of them "
                    "carried stage names or touchpoints in any recognized shape.",
                    field="cx.journeys[].stages[].touchpoints",
                    inputs={"journey_count": len(journeys)},
                ),
            )

        conf_label = (
            tenant_profile.cx_journeys_confidence
            or tenant_profile.cx_confidence
        )
        conf_base = {"High": 0.90, "Medium": 0.80, "Low": 0.65}.get(conf_label, 0.70)

        return TagResult(
            value=touchpoints,
            source="deterministic",
            confidence=conf_base,
            evidence=ev.profile(
                "tenant.cx_touchpoints.extracted",
                f"Collected {len(touchpoints)} distinct touchpoint(s) from "
                f"{len(journeys)} CX journey(s) in the tenant profile; confidence "
                f"tracks the agent's own rating of that section "
                f"({conf_label or 'unrated'}).",
                field="cx.journeys[].stages[].touchpoints",
                inputs={"journey_count": len(journeys),
                        "touchpoint_count": len(touchpoints),
                        "agent_confidence": conf_label or "unknown"},
                quote=", ".join(touchpoints[:8]),
            ),
        )


def create_tagger() -> KeyCxTouchpointsTagger:
    return KeyCxTouchpointsTagger()
