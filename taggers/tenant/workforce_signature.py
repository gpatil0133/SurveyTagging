"""Workforce signature tagger.

Combines tenant_profile.frontline_ratio, work_arrangement, and workforce_composition
into a single coarse signature. Frontline-heavy tenants need different default
dashboards (operational, manager-effectiveness) than knowledge-worker tenants
(engagement, culture).

Allowed values: Frontline-heavy / Knowledge / Hybrid / Field / Distributed / N/A
"""

from __future__ import annotations

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
                evidence="No tenant_profile EX data available",
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
                evidence=f"frontline_ratio={frontline!r}, work_arrangement={arrangement!r}",
            )

        # Otherwise honor explicit work arrangement signals
        if arr:
            return TagResult(
                value=arr,
                source="deterministic",
                confidence=conf_base,
                evidence=f"work_arrangement={arrangement!r}, frontline_ratio={frontline!r}",
            )

        # Composition hints
        composition_lower = composition.lower()
        if band == "medium":
            return TagResult(
                value="Hybrid",
                source="deterministic",
                confidence=conf_base * 0.9,
                evidence=f"Mixed frontline+knowledge — frontline_ratio={frontline!r}, composition={composition!r}",
            )
        if "knowledge" in composition_lower or "professional" in composition_lower or band == "low":
            return TagResult(
                value="Knowledge",
                source="deterministic",
                confidence=conf_base,
                evidence=f"composition={composition!r}, frontline_ratio={frontline!r}",
            )

        return TagResult(
            value="N/A",
            source="deterministic",
            confidence=0.50,
            evidence=(
                f"Unrecognized signals — composition={composition!r}, "
                f"arrangement={arrangement!r}, frontline={frontline!r}"
            ),
        )


def create_tagger() -> WorkforceSignatureTagger:
    return WorkforceSignatureTagger()
