"""Topic/theme tagger: LLM-based with keyword priors."""

from __future__ import annotations

import re

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

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped")

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
                        evidence="Demographic question content + role",
                    )

        # Keyword prior matching
        for keywords, theme in _TOPIC_KEYWORDS:
            if any(kw in combined for kw in keywords):
                return TagResult(
                    value=theme,
                    source="hybrid",
                    confidence=0.75,
                    evidence=f"Keyword match for {theme}",
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
                evidence="Service/support keywords in question",
            )

        # Product Experience: questions about product quality
        product_keywords = ["product", "feature", "quality", "design", "usability", "functionality",
                            "performance", "reliability"]
        if any(kw in combined for kw in product_keywords):
            return TagResult(
                value="Product Experience",
                source="hybrid",
                confidence=0.70,
                evidence="Product experience keywords",
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
            evidence="Requires LLM classification",
        )


def create_tagger() -> TopicThemeTagger:
    return TopicThemeTagger()
