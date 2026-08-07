"""Role/intent tagger: hybrid deterministic (phase 1) + LLM (phase 2)."""

from __future__ import annotations

import re

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

_DEMOGRAPHIC_PATTERNS = [
    re.compile(r"\bage\s*(group|range)?\b", re.I),
    re.compile(r"\bgender\b", re.I),
    re.compile(r"\bethnicity\b", re.I),
    re.compile(r"\beducation\s*level\b", re.I),
    re.compile(r"\bmarital\s*status\b", re.I),
    re.compile(r"\bdate\s*of\s*birth\b", re.I),
    re.compile(r"\bfull\s*name\b", re.I),
    re.compile(r"^name$", re.I),
    re.compile(r"\bemail\s*address\b", re.I),
    re.compile(r"\bcity\s*/?\s*town\b", re.I),
    re.compile(r"\bphone\s*(number)?\b", re.I),
]

_DEMOGRAPHIC_OPTION_KEYWORDS = [
    "male", "female", "non-binary", "prefer not",
    "18-24", "25-34", "35-44", "45-54", "55-64", "65",
    "employed", "unemployed", "retired", "student", "self-employed",
]


class RoleIntentTagger(QuestionTagger):
    name = "question.role_intent"
    tag_dimension = "role_intent"
    stage = 3
    source_type = "hybrid"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Content message, not a question")

        q = question

        # Primary Metric: NPS
        if q.rs_type == 2:
            return TagResult(value="Primary Metric", source="deterministic", confidence=1.0,
                             evidence="rsType=2 (NPS 0-10)")

        # Primary Metric: CSAT
        if q.rs_type == 4:
            return TagResult(value="Primary Metric", source="deterministic", confidence=1.0,
                             evidence="rsType=4 (CSAT 5-point)")

        # Primary Metric: Custom metric
        if q.is_custom_metric:
            return TagResult(value="Primary Metric", source="deterministic", confidence=0.95,
                             evidence=f"isCustomMetric=true ({q.custom_metric_title})")

        # Follow-up / Verbatim
        if q.is_followup_question and q.question_type == "T":
            return TagResult(value="Follow-up / Verbatim", source="deterministic", confidence=0.95,
                             evidence=f"Follow-up to question {q.metric_question_id}")

        # Profiling / Demographic: email subtype
        if q.question_type == "T" and q.question_sub_type == 31:
            return TagResult(value="Profiling / Demographic", source="deterministic", confidence=0.95,
                             evidence="Email validation subType=31")

        # Profiling / Demographic: contact/signature
        if q.question_type == "CS":
            return TagResult(value="Profiling / Demographic", source="deterministic", confidence=0.95,
                             evidence="Contact/Signature question type")

        # Profiling / Demographic: title matches demographic patterns
        if any(p.search(q.title) for p in _DEMOGRAPHIC_PATTERNS):
            return TagResult(value="Profiling / Demographic", source="deterministic", confidence=0.85,
                             evidence="Title matches demographic pattern")

        # Profiling / Demographic: answer options are demographic
        if q.question_type in ("L", "R", "RT") and q.answer_options:
            opt_text = " ".join(o.answer_text.lower() for o in q.answer_options)
            demo_matches = sum(1 for kw in _DEMOGRAPHIC_OPTION_KEYWORDS if kw in opt_text)
            if demo_matches >= 3:
                return TagResult(value="Profiling / Demographic", source="deterministic", confidence=0.80,
                                 evidence=f"Answer options match demographic patterns ({demo_matches} matches)")

        # Contextual / Situational: date picker
        if q.question_type == "T" and q.question_sub_type == 1:
            return TagResult(value="Contextual / Situational", source="deterministic", confidence=0.85,
                             evidence="Date picker subType=1")

        # Contextual / Situational: file upload
        if q.question_type == "T" and q.question_sub_type == 71:
            return TagResult(value="Contextual / Situational", source="deterministic", confidence=0.80,
                             evidence="File upload subType=71")

        # Segmentation: hidden radio (routing)
        if q.question_type == "HR":
            return TagResult(value="Segmentation", source="deterministic", confidence=0.85,
                             evidence="Hidden radio (routing question)")

        # Primary Metric: CES (Customer Effort Score) — rs_type=3 per Sogolytics platform
        if q.rs_type == 3:
            return TagResult(value="Primary Metric", source="deterministic", confidence=1.0,
                             evidence="rsType=3 (CES effort scale)")

        # Driver/Attribute: Key driver flag
        if q.is_key_driver:
            return TagResult(value="Driver / Attribute", source="deterministic", confidence=0.90,
                             evidence="isKeyDriver=true")

        # Driver/Attribute: matrix/grid types
        if q.question_type in ("RW", "RK", "GR", "GC", "RG", "GQ"):
            return TagResult(value="Driver / Attribute", source="deterministic", confidence=0.80,
                             evidence=f"Matrix/grid question type: {q.question_type}")

        # Segmentation: L/R/RT with categorical options (not rating)
        if q.question_type in ("L", "R", "SR", "ML", "C"):
            # If it has weights, it's more like a rating
            has_weights = any(o.weight is not None for o in q.answer_options)
            if not has_weights:
                return TagResult(value="Segmentation", source="deterministic", confidence=0.70,
                                 evidence=f"Categorical {q.question_type} without weights")

        # RT with weights = rating
        if q.question_type == "RT":
            return TagResult(value="Driver / Attribute", source="deterministic", confidence=0.75,
                             evidence="Visual rating question")

        # RS (generic rating scale)
        if q.question_type == "RS":
            return TagResult(value="Driver / Attribute", source="deterministic", confidence=0.75,
                             evidence="Rating scale question")

        # Open-ended text: verbatim/follow-up or diagnostic
        if q.question_type == "T":
            # Check if it's at the end (closing question)
            if q.effective_position_ratio > 0.85:
                return TagResult(value="Follow-up / Verbatim", source="hybrid", confidence=0.70,
                                 evidence="Open-ended text near end of survey")
            return TagResult(value="Diagnostic", source="hybrid", confidence=0.60,
                             evidence="Open-ended text (requires LLM refinement)")

        # Fallback
        return TagResult(value="Segmentation", source="hybrid", confidence=0.50,
                         evidence="Requires LLM classification")


def create_tagger() -> RoleIntentTagger:
    return RoleIntentTagger()
