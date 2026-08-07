"""metric_name tagger — assigns a named metric (NPS, CSAT, CES, eNPS, or derived custom).

Stage 3, hybrid. Depends on project `project_type` tag (Stage 1) for eNPS detection.
Cardinality controlled via 3-word cap + HTML cleaning.
"""

from __future__ import annotations

import re

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
                             evidence="Content message")

        # NPS vs eNPS — depends on project project_type (Stage 1, already populated)
        if q.rs_type == 2:
            project_type = accumulator.get_project_tag_value("project_type")
            if project_type == "EX":
                return TagResult(value="eNPS", source="deterministic",
                                 confidence=1.0, evidence="rs_type=2 + project_type=EX")
            return TagResult(value="NPS", source="deterministic",
                             confidence=1.0, evidence="rs_type=2 (NPS)")

        # CES — always rs_type=3 per platform
        if q.rs_type == 3:
            return TagResult(value="CES", source="deterministic",
                             confidence=1.0, evidence="rs_type=3 (CES)")

        # CSAT
        if q.rs_type == 4:
            return TagResult(value="CSAT", source="deterministic",
                             confidence=1.0, evidence="rs_type=4 (CSAT)")

        # Custom metric — use custom_metric_title (cleaned)
        if q.is_custom_metric and q.custom_metric_title:
            cleaned = _clean_and_shorten(q.custom_metric_title, max_words=4)
            if cleaned:
                return TagResult(value=cleaned, source="deterministic",
                                 confidence=0.95,
                                 evidence=f"custom_metric_title: {q.custom_metric_title!r}")

        # Open-ended
        if q.question_type == "T":
            return TagResult(value="Text Feedback", source="deterministic",
                             confidence=0.95, evidence="Text question")

        # Matrix grid — use matrix_group_title (cleaned)
        if q.matrix_group_title:
            cleaned = _clean_and_shorten(q.title, max_words=3)
            if cleaned:
                return TagResult(value=cleaned, source="hybrid",
                                 confidence=0.70,
                                 evidence=f"Matrix row title (grid: {q.matrix_group_title!r})")

        # Generic rating scale — derive from title
        if q.question_type in ("RS", "RT", "RW", "RK", "RG"):
            fp = q.scale_fingerprint or ""
            if "satisfaction" in fp:
                return TagResult(value="Overall Satisfaction", source="hybrid",
                                 confidence=0.75, evidence=f"satisfaction pattern: {fp}")
            derived = _clean_and_shorten(q.title, max_words=3)
            if derived:
                return TagResult(value=derived + " Rating", source="hybrid",
                                 confidence=0.55,
                                 evidence="Derived from title")

        # Categorical or unclassified — no standard name
        return TagResult(value=None, source="deterministic", status="skipped",
                         evidence="Not a measurable metric")


def create_tagger() -> MetricNameTagger:
    return MetricNameTagger()
