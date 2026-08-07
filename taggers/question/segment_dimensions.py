"""segment_dimensions tagger — multi-label list of dimensions this question can segment by.

Stage 4, multi_label, deterministic. Depends on is_segmentable (Stage 4).

Returns [] (empty list, NOT skipped) for non-segmentable questions to keep
the tag present in the output schema.
"""

from __future__ import annotations

import re

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


# Title keyword -> canonical dimension name
_TITLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdepartment\b|\bdept\b|\bteam\b", re.I), "Department"),
    (re.compile(r"\bregion\b|\bterritory\b", re.I), "Region"),
    (re.compile(r"\bcountry\b|\bnation\b", re.I), "Country"),
    (re.compile(r"\bstate\b|\bprovince\b", re.I), "State"),
    (re.compile(r"\bcity\b|\btown\b", re.I), "City"),
    (re.compile(r"\bbranch\b|\boffice\b|\blocation\b", re.I), "Location"),
    (re.compile(r"\bage\b|\bage\s*group\b|\bage\s*range\b", re.I), "Age"),
    (re.compile(r"\bgender\b|\bsex\b", re.I), "Gender"),
    (re.compile(r"\brole\b|\bposition\b|\bjob\s*title\b", re.I), "Role"),
    (re.compile(r"\btenure\b|\byears?\s*of\s*service\b", re.I), "Tenure"),
    (re.compile(r"\bproduct\b|\bsku\b|\bbrand\b", re.I), "Product"),
    (re.compile(r"\bcustomer\s*type\b|\bsegment\b", re.I), "Customer Type"),
    (re.compile(r"\bindustry\b|\bvertical\b", re.I), "Industry"),
    (re.compile(r"\bcompany\s*size\b|\bheadcount\b", re.I), "Company Size"),
    (re.compile(r"\btier\b|\bplan\b|\bsubscription\b", re.I), "Plan/Tier"),
    (re.compile(r"\bchannel\b", re.I), "Channel"),
    (re.compile(r"\bgrade\b|\bclass\b", re.I), "Grade"),
    (re.compile(r"\beducation\s*level\b", re.I), "Education Level"),
    (re.compile(r"\bethnicity\b|\brace\b", re.I), "Ethnicity"),
    (re.compile(r"\bmarital\s*status\b", re.I), "Marital Status"),
    (re.compile(r"\bemployment\s*status\b", re.I), "Employment Status"),
]

# Options-based heuristics (if the option texts look demographic)
_OPTION_KEYWORDS_TO_DIM: dict[str, str] = {
    "male": "Gender", "female": "Gender", "non-binary": "Gender",
    "18-24": "Age", "25-34": "Age", "35-44": "Age", "45-54": "Age",
    "55-64": "Age", "65+": "Age",
    "employed": "Employment Status", "unemployed": "Employment Status",
    "student": "Employment Status", "retired": "Employment Status",
    "single": "Marital Status", "married": "Marital Status",
    "urban": "Location", "suburban": "Location", "rural": "Location",
}


class SegmentDimensionsTagger(QuestionTagger):
    name = "question.segment_dimensions"
    tag_dimension = "segment_dimensions"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.is_segmentable"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=[], source="deterministic", status="skipped",
                             evidence="Content message")

        segmentable = accumulator.get_question_tag_value(q.question_id, "is_segmentable")
        if segmentable != "Yes":
            return TagResult(value=[], source="deterministic", confidence=1.0,
                             evidence="Not segmentable")

        dims: set[str] = set()

        # Title-based extraction
        for pattern, dim in _TITLE_PATTERNS:
            if pattern.search(q.title):
                dims.add(dim)

        # Answer-option extraction
        if q.answer_options:
            opt_text = " ".join(o.answer_text.lower() for o in q.answer_options if o.answer_text)
            for kw, dim in _OPTION_KEYWORDS_TO_DIM.items():
                if kw in opt_text:
                    dims.add(dim)

        if dims:
            return TagResult(value=sorted(dims), source="deterministic",
                             confidence=0.85, evidence=f"Extracted from title/options")

        # Segmentable but dimension not inferred — return empty list, not skip
        return TagResult(value=[], source="deterministic", confidence=0.60,
                         evidence="Segmentable but dimension unclear")


def create_tagger() -> SegmentDimensionsTagger:
    return SegmentDimensionsTagger()
