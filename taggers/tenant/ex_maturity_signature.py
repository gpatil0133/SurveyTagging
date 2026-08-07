"""EX maturity signature tagger.

Normalizes tenant_profile.ex_maturity_level to a canonical 4-tier signature.
Backend uses this to decide dashboard complexity — Foundational tenants get
simple 3-widget Executive Dashboards; Advanced get full driver/cross-tab
overlays.

Allowed values: Foundational / Developing / Established / Advanced / N/A
"""

from __future__ import annotations

from models.tags import TagResult
from models.tenant_profile import TenantProfile
from taggers.tenant.base import TenantTagger


_MATURITY_KEYWORDS: list[tuple[str, str]] = [
    ("advanced", "Advanced"),
    ("mature", "Advanced"),
    ("optimized", "Advanced"),
    ("established", "Established"),
    ("intermediate", "Established"),
    ("developing", "Developing"),
    ("emerging", "Developing"),
    ("growing", "Developing"),
    ("foundational", "Foundational"),
    ("basic", "Foundational"),
    ("initial", "Foundational"),
    ("nascent", "Foundational"),
]


class ExMaturitySignatureTagger(TenantTagger):
    name = "tenant.ex_maturity_signature"
    tag_dimension = "ex_maturity_signature"
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

        raw = tenant_profile.ex_maturity_level or ""
        if not raw:
            return TagResult(
                value="N/A",
                source="deterministic",
                confidence=0.50,
                evidence="ex_maturity.level absent from profile",
            )

        conf_base = {"High": 0.95, "Medium": 0.85, "Low": 0.65}.get(
            tenant_profile.ex_lifecycle_confidence or tenant_profile.ex_confidence,
            0.75,
        )

        lower = raw.lower()
        for needle, canon in _MATURITY_KEYWORDS:
            if needle in lower:
                return TagResult(
                    value=canon,
                    source="deterministic",
                    confidence=conf_base,
                    evidence=f"ex_maturity.level={raw!r}",
                )

        return TagResult(
            value="N/A",
            source="deterministic",
            confidence=0.50,
            evidence=f"Unrecognized maturity label: {raw!r}",
        )


def create_tagger() -> ExMaturitySignatureTagger:
    return ExMaturitySignatureTagger()
