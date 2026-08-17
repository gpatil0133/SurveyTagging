"""Industry vertical tagger: tenant profile verbatim, else a seed the LLM rewrites.

`industry_vertical` is free text (user_defined). The ordering below follows from
that: the org research agent already writes a precise label and it is now stored
in its own words, so it goes first and sits above the 0.80 threshold at which
`_apply_project_llm_results` declines to override a non-LLM tag. Every other tier
is a guess about a survey rather than knowledge about a tenant, so each is held
below that line and LLM Call 1 replaces it.

Values are no longer coerced onto a ten-item enum. `_normalize_agent_industry`
survives as a *derived* lookup key (journey_stages.yaml templates key on the short
names) — never as the stored tag.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

# Keep a profile-sourced value clear of the LLM override threshold in
# `pipeline/llm_enhance._apply_project_llm_results` (>= 0.80 and non-LLM wins).
_PROFILE_CONFIDENCE = {"High": 0.95, "Medium": 0.90}
_PROFILE_CONFIDENCE_DEFAULT = 0.85

# Everything below the profile is a seed. Held under 0.80 on purpose so LLM
# Call 1 rewrites it — "no profile -> let the model decide" is the whole rule.
_DIRECTORY_CONFIDENCE = 0.75
_OVERRIDE_CONFIDENCE = 0.70
_KEYWORD_CONFIDENCE_BASE = 0.55
_KEYWORD_CONFIDENCE_CAP = 0.75


def _profile_label(profile) -> str:
    """The agent's industry in its own words, sub-vertical appended when distinct.

    "Financial Services" + "Regional Retail Banking" reads better as one string
    than either does alone, and the enum this replaced could hold neither.
    """
    primary = (profile.industry_vertical or "").strip()
    sub = (profile.industry_sub_vertical or "").strip()
    if not primary:
        return ""
    if sub and sub.lower() != primary.lower():
        return f"{primary} / {sub}"
    return primary


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
        # Tier 1: the tenant profile, VERBATIM. The org research agent read the
        # tenant's own site to write this; the tiers below only read one survey.
        # Stored above 0.80 so LLM Call 1 keeps its hands off it.
        profile = context.tenant_profile
        if profile is not None and profile.has_org:
            label = _profile_label(profile)
            if label:
                conf = _PROFILE_CONFIDENCE.get(
                    profile.org_confidence, _PROFILE_CONFIDENCE_DEFAULT
                )
                return TagResult(
                    value=label,
                    source="hybrid",
                    confidence=conf,
                    evidence=ev.profile(
                        "project.industry.tenant_profile",
                        f'The org research agent reports "{label}", and that is stored '
                        "as written rather than snapped to a short list — the agent "
                        "researched this tenant, while every other tier here is "
                        "inferring an industry from one survey. Confidence tracks the "
                        f"agent's own rating ({profile.org_confidence or 'unrated'}) "
                        "but stays above the threshold at which the LLM pass would "
                        "overwrite it.",
                        field="org.industry_vertical",
                        stage=1,
                        inputs={"primary": profile.industry_vertical,
                                "sub_vertical": profile.industry_sub_vertical or "(none)",
                                "agent_confidence": profile.org_confidence or "unknown"},
                        quote=profile.industry_vertical,
                    ),
                )

        # Tier 2: Directory schema signals — the columns the customer configured
        # in their respondent directory. A real signal, but a coarse bucket name
        # rather than a description, so it seeds and the LLM may replace it.
        dir_signals = context.directory_signals
        if dir_signals.inferred_domains:
            top_domain = dir_signals.inferred_domains[0]
            return TagResult(
                value=top_domain,
                source="hybrid",
                confidence=_DIRECTORY_CONFIDENCE,
                evidence=ev.rule(
                    "project.industry.directory_schema",
                    f"No tenant profile, so the respondent directory decides: its own "
                    f"column names point at {top_domain}. The customer configured those "
                    "fields to hold real data, which makes this a solid seed — but it "
                    "is a bucket name, not a description of the business, so it is held "
                    "below the LLM override threshold.",
                    stage=1,
                    inputs={"matched_domain": top_domain,
                            "domain_keywords": dir_signals.domain_keywords[:5],
                            "all_inferred_domains": dir_signals.inferred_domains[:3]},
                ),
            )

        # Tier 3: caller-supplied industry override (ad-hoc /api/tag only).
        # Free text now, so the caller's own wording is kept; _INDUSTRY_MAP only
        # expands the handful of shorthands the ad-hoc form offers.
        override_industry = context.overrides.industry.strip()
        if override_industry and override_industry != "--":
            mapped = _INDUSTRY_MAP.get(override_industry, override_industry)
            return TagResult(
                value=mapped,
                source="hybrid",
                confidence=_OVERRIDE_CONFIDENCE,
                evidence=ev.rule(
                    "project.industry.caller_override",
                    f'The caller supplied industry "{override_industry}" on the '
                    "request. Reached only when neither the tenant profile nor the "
                    "directory produced a domain.",
                    stage=1,
                    inputs={"override": override_industry, "stored_as": mapped},
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
                confidence=min(_KEYWORD_CONFIDENCE_BASE + best_score * 0.05,
                               _KEYWORD_CONFIDENCE_CAP),
                evidence=ev.statistic(
                    "project.industry.content_keywords",
                    f"Last resort before defaulting: the survey title and question text "
                    f"contain {best_score} {best_match} keyword(s), more than any other "
                    "vertical scored. Confidence rises with the hit count but is capped "
                    "below the LLM override threshold — counting words in one survey is "
                    "a weaker read of a tenant's industry than the model's.",
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
                "All four tiers came up empty: no usable tenant profile industry, no "
                "directory domain columns, no caller override, and fewer than two "
                "vertical keywords anywhere in the survey text. Other is a placeholder "
                "and the 0.40 confidence invites the LLM pass to write the real one.",
                stage=1,
                inputs={"best_keyword_domain": best_match or "(none)",
                        "best_keyword_score": best_score},
            ),
        )


def create_tagger() -> IndustryTagger:
    return IndustryTagger()
