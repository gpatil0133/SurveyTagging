"""EX maturity signature tagger.

Normalizes tenant_profile.ex_maturity_level to a canonical 4-tier signature.
Backend uses this to decide dashboard complexity — Foundational tenants get
simple 3-widget Executive Dashboards; Advanced get full driver/cross-tab
overlays.

Allowed values: Foundational / Developing / Established / Advanced / N/A
"""

from __future__ import annotations

from models import evidence as ev
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
                evidence=ev.fallback(
                    "tenant.ex_maturity.no_ex_profile",
                    "No EX intelligence artifact on disk, so there was no maturity "
                    "statement to normalize. Downstream should read N/A as unknown "
                    "and fall back to a simple dashboard.",
                    inputs={"has_ex": False},
                ),
            )

        raw = tenant_profile.ex_maturity_level or ""
        if not raw:
            return TagResult(
                value="N/A",
                source="deterministic",
                confidence=0.50,
                evidence=ev.profile(
                    "tenant.ex_maturity.field_absent",
                    "The EX profile exists but its maturity level field is empty — the "
                    "agent did not report one.",
                    field="ex.ex_maturity.level",
                ),
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
                    evidence=ev.profile(
                        "tenant.ex_maturity.keyword_match",
                        f"The profile's maturity statement contains \"{needle}\", which "
                        f"this tagger maps to the {canon} tier.",
                        field="ex.ex_maturity.level",
                        inputs={"matched_keyword": needle, "mapped_tier": canon},
                        quote=raw,
                    ),
                )

        return TagResult(
            value="N/A",
            source="deterministic",
            confidence=0.50,
            evidence=ev.profile(
                "tenant.ex_maturity.unrecognized_label",
                "The profile reports a maturity level, but its wording matched none of "
                "the Foundational / Developing / Established / Advanced keywords.",
                field="ex.ex_maturity.level",
                quote=raw,
            ),
        )


def create_tagger() -> ExMaturitySignatureTagger:
    return ExMaturitySignatureTagger()
