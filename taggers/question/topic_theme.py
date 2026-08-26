"""Topic/theme tagger: LLM-based with keyword priors."""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# Keyword priors for pre-LLM classification
_TOPIC_KEYWORDS: list[tuple[list[str], str]] = [
    (["manager", "leadership", "supervisor", "boss", "direct report"], "Manager / Leadership"),
    (["salary", "compensation", "benefits", "perks", "pay", "bonus"], "Compensation & Benefits"),
    (["culture", "belonging", "diversity", "inclusion", "dei", "psychological safety"], "Culture & Belonging"),
    (["onboarding", "training", "orientation", "learning", "development"], "Onboarding"),
    (["price", "pricing", "value", "cost", "affordable", "expensive"], "Pricing & Value"),
    (["competitor", "alternative", "compared to", "versus", "other brands"], "Competitor / Market"),
    (["brand", "reputation", "awareness", "image", "recommend", "nps", "promoter"], "Brand & Perception"),
]

_DEMOGRAPHIC_PATTERNS = [
    re.compile(r"\bage\b|\bgender\b|\bincome\b|\bemployment\b|\bregion\b|\bname\b|\bemail\b", re.I),
]


class TopicThemeTagger(QuestionTagger):
    name = "question.topic_theme"
    tag_dimension = "topic_theme"
    stage = 5
    source_type = "llm"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        title_lower = question.title.lower()
        group_lower = question.matrix_group_title.lower()
        combined = f"{title_lower} {group_lower}"

        # Demographics check
        for pattern in _DEMOGRAPHIC_PATTERNS:
            if pattern.search(combined):
                role = accumulator.get_question_tag_value(question.question_id, "role_intent")
                if role in ("Profiling / Demographic", "Segmentation"):
                    return TagResult(
                        value="Demographics",
                        source="hybrid",
                        confidence=0.85,
                        evidence=ev.hybrid(
                            "question.topic_theme.demographics",
                            "Two signals agree: the wording matches a demographic "
                            "pattern (age / gender / income / region / name / email) "
                            f"and role_intent is already {role}. Either alone would be "
                            "weaker — plenty of non-demographic questions mention age.",
                            components=[
                                ev.component("demographic keyword",
                                             f"matched {pattern.pattern!r}"),
                                ev.component("role_intent", role),
                            ],
                            stage=5,
                        ),
                    )

        # Keyword prior matching
        for keywords, theme in _TOPIC_KEYWORDS:
            if any(kw in combined for kw in keywords):
                matched = [kw for kw in keywords if kw in combined]
                return TagResult(
                    value=theme,
                    source="hybrid",
                    confidence=0.75,
                    evidence=ev.rule(
                        "question.topic_theme.keyword_prior",
                        f"The question or its matrix stem contains "
                        f"{matched[0]!r}, which this tagger maps to {theme}. A prior "
                        "only — the Stage 5 LLM pass may still overrule it.",
                        stage=5,
                        inputs={"matched_keywords": matched[:3], "theme": theme},
                        quote=question.title,
                    ),
                )

        # Contextual inference from survey industry
        industry = accumulator.get_project_tag_value("industry_vertical")
        purpose = accumulator.get_project_tag_value("project_purpose")

        # Service/Support: questions about interaction quality in service contexts
        service_keywords = ["service", "support", "staff", "agent", "communication", "response time",
                            "help desk", "interaction", "assist"]
        if any(kw in combined for kw in service_keywords):
            return TagResult(
                value="Service / Support",
                source="hybrid",
                confidence=0.75,
                evidence=ev.rule(
                    "question.topic_theme.service_keywords",
                    "The question asks about an interaction with people or a support "
                    "channel (service, staff, agent, response time...) rather than "
                    "about a product itself.",
                    stage=5,
                    inputs={"matched_keywords":
                                [kw for kw in service_keywords if kw in combined][:3]},
                    quote=question.title,
                ),
            )

        # Product Experience: questions about product quality
        product_keywords = ["product", "feature", "quality", "design", "usability", "functionality",
                            "performance", "reliability"]
        if any(kw in combined for kw in product_keywords):
            return TagResult(
                value="Product Experience",
                source="hybrid",
                confidence=0.70,
                evidence=ev.rule(
                    "question.topic_theme.product_keywords",
                    "The question asks about the thing itself — its quality, features, "
                    "design or reliability — rather than about the people delivering "
                    "it.",
                    stage=5,
                    inputs={"matched_keywords":
                                [kw for kw in product_keywords if kw in combined][:3]},
                    quote=question.title,
                ),
            )

        # Default — will be refined by LLM
        if purpose == "Employee Engagement":
            default = "Culture & Belonging"
        elif industry in ("Healthcare", "Hospitality / Travel"):
            default = "Service / Support"
        else:
            default = "Product Experience"

        return TagResult(
            value=default,
            source="llm",
            confidence=0.40,
            evidence=ev.fallback(
                "question.topic_theme.deferred_to_llm",
                f"No keyword prior matched, so the theme was guessed from survey "
                f"context alone: purpose {purpose or 'unset'} and industry "
                f"{industry or 'unset'} make {default} the likeliest default. Held at "
                "0.40 so LLM Call 2 replaces it — if you see this in the output, that "
                "call did not answer for this question.",
                stage=5,
                inputs={"project_purpose": purpose or "(unset)",
                        "industry_vertical": industry or "(unset)",
                        "defaulted_to": default},
            ),
        )


def create_tagger() -> TopicThemeTagger:
    return TopicThemeTagger()
