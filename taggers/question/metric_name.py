"""metric_name tagger — assigns a named metric (NPS, CSAT, CES, eNPS, or derived custom).

Stage 3, hybrid. Depends on project `project_type` tag (Stage 1) for eNPS detection.
Cardinality controlled via 3-word cap + HTML cleaning.
"""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "how", "what", "is", "are", "your", "you", "the", "a", "an", "of",
    "with", "on", "in", "at", "to", "from", "for", "and", "or", "do", "does",
    "rate", "please", "was", "were", "our", "this", "that", "would", "we",
    "overall", "likely", "satisfied", "not", "at", "all",
}


def _clean_and_shorten(text: str, max_words: int = 3) -> str:
    """Strip HTML, normalize whitespace, title-case, cap word count."""
    if not text:
        return ""
    t = _HTML_ENTITY_RE.sub(" ", text)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    # Remove trailing ?/.
    t = t.rstrip("?.!").strip()
    words = t.split()
    content_words = [w for w in words if w.lower() not in _STOPWORDS]
    if not content_words:
        content_words = words
    short = " ".join(content_words[:max_words])
    # Title case each word except uppercase acronyms
    return " ".join(w if w.isupper() else w.capitalize() for w in short.split())


class MetricNameTagger(QuestionTagger):
    name = "question.metric_name"
    tag_dimension = "metric_name"
    stage = 3
    source_type = "hybrid"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("metric_name", stage=3))

        # NPS vs eNPS — depends on project project_type (Stage 1, already populated)
        if q.rs_type == 2:
            project_type = accumulator.get_project_tag_value("project_type")
            if project_type == "EX":
                return TagResult(
                    value="eNPS", source="deterministic", confidence=1.0,
                    evidence=ev.rule(
                        "question.metric_name.enps",
                        "The platform flags this as a Net Promoter question "
                        "(rs_type=2) and the survey is typed EX, so the respondents "
                        "are employees — the same scale reported as eNPS rather than "
                        "NPS, and benchmarked against a different population.",
                        stage=3,
                        inputs={"rs_type": 2, "project_type": "EX"},
                    ),
                )
            return TagResult(
                value="NPS", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_name.nps",
                    "The platform flags this as a Net Promoter question (rs_type=2) "
                    "and the survey is not an EX one, so it is customer NPS.",
                    stage=3,
                    inputs={"rs_type": 2,
                            "project_type": project_type or "(unset)"},
                ),
            )

        # CES — always rs_type=3 per platform
        if q.rs_type == 3:
            return TagResult(
                value="CES", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_name.ces",
                    "The platform flags this as a Customer Effort Score question "
                    "(rs_type=3).",
                    stage=3,
                    inputs={"rs_type": 3},
                ),
            )

        # CSAT
        if q.rs_type == 4:
            return TagResult(
                value="CSAT", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_name.csat",
                    "The platform flags this as a Customer Satisfaction question "
                    "(rs_type=4).",
                    stage=3,
                    inputs={"rs_type": 4},
                ),
            )

        # Custom metric — use custom_metric_title (cleaned)
        if q.is_custom_metric and q.custom_metric_title:
            cleaned = _clean_and_shorten(q.custom_metric_title, max_words=4)
            if cleaned:
                return TagResult(
                    value=cleaned, source="deterministic", confidence=0.95,
                    evidence=ev.rule(
                        "question.metric_name.custom_metric_title",
                        "The survey author named this custom metric themselves; the "
                        "name here is their title with HTML stripped and capped at "
                        "four words to keep tag cardinality manageable.",
                        stage=3,
                        inputs={"is_custom_metric": True, "shortened_to": cleaned},
                        quote=q.custom_metric_title,
                    ),
                )

        # Open-ended
        if q.question_type == "T":
            return TagResult(
                value="Text Feedback", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.metric_name.text_feedback",
                    "A free-text question has no metric name of its own — every "
                    "open-end is grouped under one label so verbatims can be found "
                    "together.",
                    stage=3,
                    inputs={"question_type": "T"},
                ),
            )

        # Matrix grid — use matrix_group_title (cleaned)
        if q.matrix_group_title:
            cleaned = _clean_and_shorten(q.title, max_words=3)
            if cleaned:
                return TagResult(
                    value=cleaned, source="hybrid", confidence=0.70,
                    evidence=ev.rule(
                        "question.metric_name.matrix_row_title",
                        "One row of a matrix, with no platform metric name. The name "
                        "is derived from the row's own text (stopwords dropped, capped "
                        "at three words) — a readable label, not an authored one, "
                        "hence 0.70.",
                        stage=3,
                        inputs={"matrix_group_title": q.matrix_group_title,
                                "derived_from": "question title"},
                        quote=q.title,
                    ),
                )

        # Generic rating scale — derive from title
        if q.question_type in ("RS", "RT", "RW", "RK", "RG"):
            fp = q.scale_fingerprint or ""
            if "satisfaction" in fp:
                return TagResult(
                    value="Overall Satisfaction", source="hybrid", confidence=0.75,
                    evidence=ev.rule(
                        "question.metric_name.satisfaction_scale",
                        "The answer scale matches a known satisfaction fingerprint "
                        "(from scale_patterns.yaml), so the question measures "
                        "satisfaction even though the platform did not flag it as CSAT.",
                        stage=3,
                        inputs={"scale_fingerprint": fp,
                                "question_type": q.question_type},
                    ),
                )
            derived = _clean_and_shorten(q.title, max_words=3)
            if derived:
                return TagResult(
                    value=derived + " Rating", source="hybrid", confidence=0.55,
                    evidence=ev.fallback(
                        "question.metric_name.derived_from_title",
                        f"A rating-scale question with no platform metric flag and no "
                        f"recognized scale fingerprint. The name is manufactured from "
                        f"the question's own wording — informative but not "
                        "authoritative, which is what the 0.55 says.",
                        stage=3,
                        inputs={"question_type": q.question_type,
                                "scale_fingerprint": fp or "(none)"},
                    ),
                )

        # Categorical or unclassified — no standard name
        return TagResult(
            value=None, source="deterministic", status="skipped",
            evidence=ev.rule(
                "question.metric_name.not_a_metric",
                f"Type {q.question_type} measures nothing — it is a categorical pick, "
                "a contact block or an unrecognized type — so there is no metric to "
                "name. Skipped rather than given a placeholder name.",
                stage=3,
                inputs={"question_type": q.question_type, "rs_type": q.rs_type,
                        "is_custom_metric": q.is_custom_metric},
            ),
        )


def create_tagger() -> MetricNameTagger:
    return MetricNameTagger()
