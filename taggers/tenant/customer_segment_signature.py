"""Customer segment signature tagger.

Normalizes tenant_profile.primary_customer_segment + secondary_customer_segments
into a coarse high-level signature the backend can act on without re-parsing
free-form labels.

Allowed values: B2B-Enterprise / B2B-SMB / B2C / B2B2C / Mixed / N/A
"""

from __future__ import annotations

from models import evidence as ev
from models.tags import TagResult
from models.tenant_profile import TenantProfile
from taggers.tenant.base import TenantTagger


def _classify(label: str) -> str | None:
    lower = label.lower()
    has_b2b = "b2b" in lower or "business" in lower or "enterprise" in lower
    has_b2c = "b2c" in lower or "consumer" in lower or "retail" in lower
    is_smb = "smb" in lower or "small" in lower or "mid-market" in lower or "mid market" in lower
    is_enterprise = "enterprise" in lower or "large" in lower
    if has_b2b and has_b2c:
        return "B2B2C"
    if has_b2b:
        if is_smb:
            return "B2B-SMB"
        if is_enterprise:
            return "B2B-Enterprise"
        return "B2B-Enterprise"
    if has_b2c:
        return "B2C"
    return None


class CustomerSegmentSignatureTagger(TenantTagger):
    name = "tenant.customer_segment_signature"
    tag_dimension = "customer_segment_signature"
    source_type = "deterministic"

    def tag(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> TagResult:
        if tenant_profile is None or not tenant_profile.has_cx:
            return TagResult(
                value="N/A",
                source="deterministic",
                confidence=0.40,
                evidence=ev.fallback(
                    "tenant.segment.no_cx_profile",
                    "No CX intelligence artifact on disk, so there was no customer-segment "
                    "statement to classify. N/A here means unknown, not 'no segment'.",
                    inputs={"has_cx": False},
                ),
            )

        primary = tenant_profile.primary_customer_segment or ""
        secondary = tenant_profile.secondary_customer_segments

        conf_base = {"High": 0.90, "Medium": 0.80, "Low": 0.65}.get(
            tenant_profile.cx_confidence, 0.70
        )

        primary_signal = _classify(primary)
        secondary_signals = {_classify(s) for s in secondary if isinstance(s, str)}
        secondary_signals.discard(None)

        all_signals = set(secondary_signals)
        if primary_signal is not None:
            all_signals.add(primary_signal)

        if len(all_signals) >= 2 and not (all_signals == {"B2B-Enterprise", "B2B-SMB"}):
            return TagResult(
                value="Mixed" if "B2B2C" not in all_signals else "B2B2C",
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.segment.multiple_signals",
                    f"The primary and secondary segment statements resolve to "
                    f"{len(all_signals)} different segment classes "
                    f"({', '.join(sorted(all_signals))}), so no single class describes "
                    "this tenant. B2B-Enterprise plus B2B-SMB alone would not count — "
                    "that is one class at two sizes.",
                    field="cx.primary_customer_segment",
                    inputs={"primary": primary or "(none)",
                            "secondary": list(secondary),
                            "resolved_classes": sorted(all_signals)},
                ),
            )

        if primary_signal:
            return TagResult(
                value=primary_signal,
                source="deterministic",
                confidence=conf_base,
                evidence=ev.profile(
                    "tenant.segment.primary",
                    f"The profile's primary customer segment classifies as "
                    f"{primary_signal}, and nothing in the secondary segments "
                    "contradicts it.",
                    field="cx.primary_customer_segment",
                    inputs={"primary": primary or "(none)"},
                    quote=primary or None,
                ),
            )

        if secondary_signals:
            value = next(iter(secondary_signals))
            return TagResult(
                value=value,
                source="deterministic",
                confidence=conf_base * 0.85,
                evidence=ev.profile(
                    "tenant.segment.secondary_only",
                    f"The primary segment statement was missing or unclassifiable, so "
                    f"{value} was inferred from the secondary segments instead — hence "
                    "the discounted confidence.",
                    field="cx.secondary_customer_segments",
                    inputs={"primary": primary or "(none)",
                            "secondary": list(secondary)},
                ),
            )

        return TagResult(
            value="N/A",
            source="deterministic",
            confidence=0.50,
            evidence=ev.profile(
                "tenant.segment.unrecognized_labels",
                "The CX profile does carry segment text, but none of it matched the "
                "B2B / B2C / enterprise / SMB vocabulary this tagger recognizes.",
                field="cx.primary_customer_segment",
                inputs={"primary": primary or "(none)",
                        "secondary": list(secondary)},
            ),
        )


def create_tagger() -> CustomerSegmentSignatureTagger:
    return CustomerSegmentSignatureTagger()
