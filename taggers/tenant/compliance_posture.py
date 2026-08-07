"""Compliance posture tagger.

Maps tenant_profile.regulatory_frameworks + regulatory_intensity to a single
high-level compliance signature. The backend uses this to decide anonymization
defaults, identifier policies, and which compliance widgets to surface.

Allowed values: None / GDPR / HIPAA / SOC2 / FERPA / PCI / Multi-Regulatory
"""

from __future__ import annotations

from models import evidence as ev
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
                evidence=ev.fallback(
                    "tenant.compliance.no_org_profile",
                    "No org profile on disk for this tenant, so no compliance signal "
                    "could be read. Defaulted to None — run the profile fetcher to improve this.",
                    inputs={"has_org": False},
                ),
            )

        frameworks = tenant_profile.regulatory_frameworks
        canonical = {_canonical_framework(f) for f in frameworks}
        canonical.discard(None)

        intensity = tenant_profile.regulatory_intensity or ""
        conf_base = {"High": 0.95, "Medium": 0.85, "Low": 0.70}.get(intensity, 0.75)

        if len(canonical) >= 2:
            named = sorted(canonical)
            return TagResult(
                value="Multi-Regulatory",
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.compliance.multi_framework",
                    f"The org profile names {len(named)} distinct regulatory frameworks "
                    f"({', '.join(named)}), which is the definition of a multi-regulatory posture.",
                    field="org.regulatory_frameworks",
                    inputs={"frameworks": named,
                            "regulatory_intensity": intensity or "unknown"},
                    quote=", ".join(frameworks),
                ),
            )
        if len(canonical) == 1:
            value = next(iter(canonical))
            return TagResult(
                value=value,
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.compliance.single_framework",
                    f"Exactly one recognized framework ({value}) appears in the org "
                    f"profile's regulatory frameworks.",
                    field="org.regulatory_frameworks",
                    inputs={"matched_framework": value,
                            "regulatory_intensity": intensity or "unknown"},
                    quote=", ".join(frameworks),
                ),
            )

        if intensity == "High":
            return TagResult(
                value="Multi-Regulatory",
                source="deterministic",
                confidence=0.55,
                evidence=ev.profile(
                    "tenant.compliance.intensity_only",
                    "None of the framework labels matched a known regime, but the profile "
                    "rates regulatory intensity as High — inferred a multi-regulatory posture "
                    "from intensity alone, hence the reduced confidence.",
                    field="org.regulatory_intensity",
                    inputs={"regulatory_intensity": "High",
                            "unmatched_labels": list(frameworks)},
                ),
            )

        return TagResult(
            value="None",
            source="deterministic",
            confidence=0.70,
            evidence=ev.profile(
                "tenant.compliance.no_framework_match",
                "No label in the org profile matched a known regulatory regime and "
                "intensity is not High, so the tenant is treated as unregulated.",
                field="org.regulatory_frameworks",
                inputs={"raw_labels": list(frameworks),
                        "regulatory_intensity": intensity or "unknown"},
            ),
        )


def create_tagger() -> CompliancePostureTagger:
    return CompliancePostureTagger()
