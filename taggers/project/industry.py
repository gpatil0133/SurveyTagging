"""Industry vertical tagger: hybrid directory signals + tenant profile + LLM."""

from __future__ import annotations

from models import evidence as ev
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
                evidence=ev.rule(
                    "project.industry.directory_schema",
                    f"The respondent directory's own column names point at "
                    f"{top_domain}. This is the strongest signal available — the "
                    "customer configured those fields themselves to hold their real "
                    "data, so it outranks both the research agent and the survey text.",
                    stage=1,
                    inputs={"matched_domain": top_domain,
                            "domain_keywords": dir_signals.domain_keywords[:5],
                            "all_inferred_domains": dir_signals.inferred_domains[:3]},
                ),
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
                    evidence=ev.profile(
                        "project.industry.tenant_profile",
                        f"No directory columns gave a domain, so the tenant profile "
                        f"decides: the org research agent reports "
                        f'"{profile.industry_vertical}", which normalizes to {mapped}. '
                        f"Confidence tracks the agent's own rating "
                        f"({profile.org_confidence or 'unrated'}) so a shaky agent "
                        "cannot outrank a caller-supplied override.",
                        field="org.industry_vertical",
                        stage=1,
                        inputs={"normalized_to": mapped,
                                "agent_confidence": profile.org_confidence or "unknown"},
                        quote=profile.industry_vertical,
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
                    evidence=ev.rule(
                        "project.industry.caller_override",
                        f'The caller supplied industry "{override_industry}" on the '
                        f"request, which maps to {mapped}. Reached only when neither "
                        "the directory nor the tenant profile produced a domain.",
                        stage=1,
                        inputs={"override": override_industry, "mapped_to": mapped},
                    ),
                )

        # Tier 4: Survey content heuristics
        #
        # NOTE: there is no corporate-record tier here. `CorporateContext.industry`
        # exists and `loaders/corporate.py::load_corporate` can read it, but
        # nothing calls that loader and `UnifiedContext` has no `corporate`
        # field — the tenant's self-reported industry never reaches a tagger.
        # Wiring it back is a deliberate decision, not a drive-by fix.
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
                evidence=ev.statistic(
                    "project.industry.content_keywords",
                    f"Last resort before defaulting: the survey title and question text "
                    f"contain {best_score} {best_match} keyword(s), more than any other "
                    "vertical scored. Confidence rises with the hit count because a "
                    "single stray word is not a vertical.",
                    measure=f"{best_match}_keyword_hits",
                    observed=best_score,
                    threshold=2,
                    stage=1,
                    inputs={"matched_domain": best_match},
                ),
            )

        # Fallback — will be refined by LLM in stage 4
        return TagResult(
            value="Other",
            source="hybrid",
            confidence=0.40,
            evidence=ev.fallback(
                "project.industry.no_signal",
                "All four tiers came up empty: no directory domain columns, no usable "
                "tenant profile industry, no caller override, and fewer than two "
                "vertical keywords anywhere in the survey text. Other is a placeholder "
                "and the 0.40 confidence invites the Stage 4 LLM pass to replace it.",
                stage=1,
                inputs={"best_keyword_domain": best_match or "(none)",
                        "best_keyword_score": best_score},
            ),
        )


def create_tagger() -> IndustryTagger:
    return IndustryTagger()
