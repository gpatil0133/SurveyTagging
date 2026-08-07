"""Industry vertical tagger: hybrid directory signals + tenant profile + LLM."""

from __future__ import annotations

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from models.tenant_profile import _normalize_agent_industry
from taggers.base import ProjectTagger


def _map_agent_industry(agent_value: str) -> str | None:
    """Coerce agent's industry string into a taxonomy value, else None.

    Thin wrapper around the shared normalizer — returns None instead of "" for
    callers that gate on truthy/None semantics.
    """
    mapped = _normalize_agent_industry(agent_value)
    return mapped or None


# Caller-supplied `industry` override → taxonomy mapping
_INDUSTRY_MAP = {
    "IT": "SaaS / Technology",
    "Information Technology": "SaaS / Technology",
    "Technology": "SaaS / Technology",
    "Healthcare": "Healthcare",
    "Health": "Healthcare",
    "Financial Services": "Financial Services",
    "Banking": "Financial Services",
    "Finance": "Financial Services",
    "Insurance": "Financial Services",
    "Education": "Higher Education",
    "Retail": "Retail / E-commerce",
    "E-commerce": "Retail / E-commerce",
    "Hospitality": "Hospitality / Travel",
    "Travel": "Hospitality / Travel",
    "Government": "Government / Public Sector",
    "Public Sector": "Government / Public Sector",
    "Fitness": "Fitness & Wellness",
    "Wellness": "Fitness & Wellness",
}


class IndustryTagger(ProjectTagger):
    name = "project.industry"
    tag_dimension = "industry_vertical"
    stage = 1
    source_type = "hybrid"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        # Tier 1: Directory schema signals (strongest — based on actual columns
        # the customer configured in their respondent directory).
        dir_signals = context.directory_signals
        if dir_signals.inferred_domains:
            top_domain = dir_signals.inferred_domains[0]
            return TagResult(
                value=top_domain,
                source="hybrid",
                confidence=0.95,
                evidence=f"Directory columns match {top_domain}: {dir_signals.domain_keywords[:5]}",
            )

        # Tier 2: Parallel.ai TenantProfile.industry_vertical (Phase 4 prior).
        # Plan flags this as "Always use" — the org agent's industry was 9/10
        # in the live audit. Confidence scales with org_confidence so a
        # Low-confidence agent doesn't override the manual-override tier.
        profile = context.tenant_profile
        if profile is not None and profile.has_org:
            mapped = _map_agent_industry(profile.industry_vertical)
            if mapped:
                conf = {"High": 0.85, "Medium": 0.75}.get(profile.org_confidence, 0.65)
                return TagResult(
                    value=mapped,
                    source="hybrid",
                    confidence=conf,
                    evidence=(
                        f"Parallel.ai org_profile.industry={profile.industry_vertical!r} "
                        f"(confidence={profile.org_confidence or 'Unknown'})"
                    ),
                )

        # Tier 3: caller-supplied industry override (ad-hoc /api/tag only)
        override_industry = context.overrides.industry.strip()
        if override_industry and override_industry != "--":
            mapped = _INDUSTRY_MAP.get(override_industry)
            if mapped:
                return TagResult(
                    value=mapped,
                    source="hybrid",
                    confidence=0.70,
                    evidence=f"Caller-supplied industry: {override_industry}",
                )

        # Tier 4: Survey content heuristics
        title_lower = context.survey_meta.title.lower()
        q_text = " ".join(q.title.lower() for q in context.questions)
        combined = f"{title_lower} {q_text}"

        content_signals = {
            "Healthcare": ["patient", "hospital", "doctor", "medical", "clinic", "opd", "nurse"],
            "Financial Services": ["bank", "account", "loan", "transaction", "financial", "credit"],
            "Higher Education": ["university", "college", "campus", "faculty", "academic"],
            "K-12 Education": ["school", "student", "teacher", "parent", "classroom", "grade"],
            "Hospitality / Travel": ["hotel", "guest", "travel", "booking", "reservation"],
            "Retail / E-commerce": ["store", "purchase", "product", "shopping", "retail"],
        }

        best_match = None
        best_score = 0
        for domain, keywords in content_signals.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_match = domain

        if best_match and best_score >= 2:
            return TagResult(
                value=best_match,
                source="hybrid",
                confidence=min(0.60 + best_score * 0.05, 0.85),
                evidence=f"Content keyword matches for {best_match} (score={best_score})",
            )

        # Fallback — will be refined by LLM in stage 4
        return TagResult(
            value="Other",
            source="hybrid",
            confidence=0.40,
            evidence="No strong industry signals detected",
        )


def create_tagger() -> IndustryTagger:
    return IndustryTagger()
