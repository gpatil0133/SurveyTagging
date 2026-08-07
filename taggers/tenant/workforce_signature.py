"""Workforce signature tagger.

Combines tenant_profile.frontline_ratio, work_arrangement, and workforce_composition
into a single coarse signature. Frontline-heavy tenants need different default
dashboards (operational, manager-effectiveness) than knowledge-worker tenants
(engagement, culture).

Allowed values: Frontline-heavy / Knowledge / Hybrid / Field / Distributed / N/A
"""

from __future__ import annotations

from models import evidence as ev
from models.tags import TagResult
from models.tenant_profile import TenantProfile
from taggers.tenant.base import TenantTagger


# Frontline ratio bands. Free-form agent output, so match on substrings.
def _frontline_band(label: str) -> str:
    lower = label.lower()
    if any(t in lower for t in ("high", ">50%", "majority", "predominant")):
        return "high"
    if any(t in lower for t in ("medium", "moderate", "mixed", "balanced")):
        return "medium"
    if any(t in lower for t in ("low", "<25%", "minimal", "small")):
        return "low"
    return ""


def _arrangement_tag(label: str) -> str:
    lower = label.lower()
    if "remote" in lower or "distributed" in lower:
        return "Distributed"
    if "field" in lower or "on-site only" in lower:
        return "Field"
    if "hybrid" in lower:
        return "Hybrid"
    return ""


class WorkforceSignatureTagger(TenantTagger):
    name = "tenant.workforce_signature"
    tag_dimension = "workforce_signature"
    source_type = "deterministic"

    def tag(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> TagResult:
        if tenant_profile is None or not tenant_profile.has_ex:
            return TagResult(
                value="N/A",
                source="deterministic",
                confidence=0.40,
                evidence=ev.fallback(
                    "tenant.workforce.no_ex_profile",
                    "No EX intelligence artifact on disk, so frontline ratio, work "
                    "arrangement and workforce composition were all unavailable. "
                    "N/A here means unknown.",
                    inputs={"has_ex": False},
                ),
            )

        frontline = tenant_profile.frontline_ratio or ""
        arrangement = tenant_profile.work_arrangement or ""
        composition = tenant_profile.workforce_composition or ""

        conf_base = {"High": 0.90, "Medium": 0.80, "Low": 0.65}.get(
            tenant_profile.ex_confidence, 0.70
        )

        band = _frontline_band(frontline)
        arr = _arrangement_tag(arrangement)

        # Frontline-heavy dominates other signals
        if band == "high":
            return TagResult(
                value="Frontline-heavy",
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.workforce.frontline_high",
                    "The EX profile describes a high frontline ratio. That outranks "
                    "work arrangement here: a mostly-frontline workforce needs "
                    "operational dashboards whether or not the office staff are hybrid.",
                    field="ex.frontline_ratio",
                    inputs={"frontline_ratio": frontline,
                            "frontline_band": "high",
                            "work_arrangement": arrangement or "(none)"},
                    quote=frontline or None,
                ),
            )

        # Otherwise honor explicit work arrangement signals
        if arr:
            return TagResult(
                value=arr,
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.workforce.arrangement",
                    f"Frontline ratio is not high, so the explicit work-arrangement "
                    f"statement decides it and reads as {arr}.",
                    field="ex.work_arrangement",
                    inputs={"work_arrangement": arrangement,
                            "frontline_ratio": frontline or "(none)",
                            "frontline_band": band or "(unmatched)"},
                    quote=arrangement or None,
                ),
            )

        # Composition hints
        composition_lower = composition.lower()
        if band == "medium":
            return TagResult(
                value="Hybrid",
                source="deterministic",
                confidence=conf_base * 0.9,
                evidence=ev.profile(
                    "tenant.workforce.frontline_medium",
                    "The frontline ratio reads as moderate/mixed and no explicit work "
                    "arrangement was given, so the workforce is treated as a "
                    "frontline-plus-knowledge blend.",
                    field="ex.frontline_ratio",
                    inputs={"frontline_ratio": frontline,
                            "frontline_band": "medium",
                            "workforce_composition": composition or "(none)"},
                ),
            )
        if "knowledge" in composition_lower or "professional" in composition_lower or band == "low":
            trigger = ("workforce_composition" if "knowledge" in composition_lower
                       or "professional" in composition_lower else "frontline_ratio")
            return TagResult(
                value="Knowledge",
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.workforce.knowledge",
                    "The workforce reads as knowledge/professional — either the "
                    "composition says so outright or the frontline ratio is low."
                    f" Deciding field here was {trigger}.",
                    field=f"ex.{trigger}",
                    inputs={"workforce_composition": composition or "(none)",
                            "frontline_ratio": frontline or "(none)",
                            "frontline_band": band or "(unmatched)"},
                ),
            )

        return TagResult(
            value="N/A",
            source="deterministic",
            confidence=0.50,
            evidence=ev.profile(
                "tenant.workforce.unrecognized_signals",
                "The EX profile exists but none of frontline ratio, work arrangement "
                "or workforce composition matched the vocabulary this tagger reads.",
                field="ex.workforce_composition",
                inputs={"workforce_composition": composition or "(none)",
                        "work_arrangement": arrangement or "(none)",
                        "frontline_ratio": frontline or "(none)"},
            ),
        )


def create_tagger() -> WorkforceSignatureTagger:
    return WorkforceSignatureTagger()
