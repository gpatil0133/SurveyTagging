"""Compliance posture tagger.

Maps tenant_profile.regulatory_frameworks + regulatory_intensity to a single
high-level compliance signature. The backend uses this to decide anonymization
defaults, identifier policies, and which compliance widgets to surface.

Allowed values: None / GDPR / HIPAA / SOC2 / FERPA / PCI / Multi-Regulatory
"""

from __future__ import annotations

from models.tags import TagResult
from models.tenant_profile import TenantProfile
from taggers.tenant.base import TenantTagger


# Free-form framework label fragments → canonical framework name. Order
# matters — first hit wins on a given fragment.
_FRAMEWORK_KEYWORDS: list[tuple[str, str]] = [
    ("hipaa", "HIPAA"),
    ("gdpr", "GDPR"),
    ("soc 2", "SOC2"),
    ("soc2", "SOC2"),
    ("soc-2", "SOC2"),
    ("ferpa", "FERPA"),
    ("pci", "PCI"),
    ("ccpa", "GDPR"),  # treat CCPA as GDPR-class privacy regime
]


def _canonical_framework(label: str) -> str | None:
    lower = label.lower()
    for needle, canon in _FRAMEWORK_KEYWORDS:
        if needle in lower:
            return canon
    return None


class CompliancePostureTagger(TenantTagger):
    name = "tenant.compliance_posture"
    tag_dimension = "compliance_posture"
    source_type = "deterministic"

    def tag(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> TagResult:
        if tenant_profile is None or not tenant_profile.has_org:
            return TagResult(
                value="None",
                source="deterministic",
                confidence=0.40,
                evidence="No tenant_profile org data available",
            )

        frameworks = tenant_profile.regulatory_frameworks
        canonical = {_canonical_framework(f) for f in frameworks}
        canonical.discard(None)

        intensity = tenant_profile.regulatory_intensity or ""
        conf_base = {"High": 0.95, "Medium": 0.85, "Low": 0.70}.get(intensity, 0.75)

        if len(canonical) >= 2:
            return TagResult(
                value="Multi-Regulatory",
                source="deterministic",
                confidence=conf_base,
                evidence=f"Multiple frameworks present: {sorted(canonical)} (intensity={intensity or 'unknown'})",
            )
        if len(canonical) == 1:
            value = next(iter(canonical))
            return TagResult(
                value=value,
                source="deterministic",
                confidence=conf_base,
                evidence=f"Framework {value} from {frameworks!r} (intensity={intensity or 'unknown'})",
            )

        if intensity == "High":
            return TagResult(
                value="Multi-Regulatory",
                source="deterministic",
                confidence=0.55,
                evidence="Regulatory intensity=High but no recognized framework labels — inferred multi-regulatory posture",
            )

        return TagResult(
            value="None",
            source="deterministic",
            confidence=0.70,
            evidence=f"No recognized frameworks (raw={frameworks!r}, intensity={intensity or 'unknown'})",
        )


def create_tagger() -> CompliancePostureTagger:
    return CompliancePostureTagger()
