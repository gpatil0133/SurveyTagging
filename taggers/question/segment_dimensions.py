"""segment_dimensions tagger — multi-label list of dimensions this question can segment by.

Stage 4, multi_label, deterministic. Depends on is_segmentable (Stage 4).

Returns [] (empty list, NOT skipped) for non-segmentable questions to keep
the tag present in the output schema.

V8 changed the SHAPE, not the labels. Entries used to be bare strings —
"Region", "Department" — which meant a dashboard composer had to resolve the
label back to a question id for every widget it segmented, once per widget.
Each entry now carries both: `{"label": "Region", "question_id": 4471}`. The
round-trip disappears and the label still reads the same to a human.

No new dimension was added for the id. That would restate what this one already
says and leave two fields to keep in agreement, which is exactly why
`control_role` was removed in V7.3.
"""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers._metric_utils import is_platform_metric
from taggers.base import QuestionTagger


# rs_type → the band a platform metric segments by. A metric is segmentable by
# its own score band, not by a respondent attribute, so it gets its own branch:
# without one, every NPS item in the corpus would land in the "segmentable but
# nothing matched — worth a human look" bucket below and drown that signal.
_METRIC_BANDS = {2: "NPS Band", 3: "CES Band", 4: "CSAT Band"}


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


def _entry(label: str, question_id: int) -> dict[str, object]:
    """One `{label, question_id}` pair.

    The id is always this question's own: `segment_dimensions` says what THIS
    question segments by, so the question a composer must send as
    `segmentationQuestion` is this one. Carrying it here is what removes the
    label-to-id lookup the composer used to do per widget.
    """
    return {"label": label, "question_id": question_id}


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
                             evidence=ev.content_message("segment_dimensions", stage=4))

        segmentable = accumulator.get_question_tag_value(q.question_id, "is_segmentable")
        if segmentable != "Yes":
            return TagResult(
                value=[], source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.segment_dimensions.not_segmentable",
                    f"is_segmentable is {segmentable or 'unset'}, so there is nothing "
                    "to enumerate. The empty list is deliberate — this dimension stays "
                    "present in the schema rather than being skipped, so consumers can "
                    "tell 'no dimensions' from 'never evaluated'.",
                    stage=4,
                    inputs={"is_segmentable": segmentable or "(unset)"},
                ),
            )

        # A platform metric segments by its own band. Handled before the title
        # patterns on purpose: a metric's title describes what is measured, not
        # an attribute of the respondent, so "how satisfied were you with your
        # branch location?" must not be read as a Location dimension.
        if is_platform_metric(q):
            band = _METRIC_BANDS.get(q.rs_type)
            if band is None:
                label = (q.custom_metric_title or "").strip()
                band = f"{label} Band" if label else "Custom Metric Band"
            return TagResult(
                value=[_entry(band, q.question_id)], source="deterministic",
                confidence=0.90,
                evidence=ev.rule(
                    "question.segment_dimensions.metric_band",
                    f"The platform scores this question, so what it segments by is its "
                    f"own score band ({band}) rather than a respondent attribute — "
                    "everything else in the survey can be compared across those bands.",
                    stage=4,
                    inputs={"rs_type": q.rs_type,
                            "is_custom_metric": q.is_custom_metric,
                            "band": band},
                ),
            )

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
            title_hits = sorted({d for p, d in _TITLE_PATTERNS if p.search(q.title)})
            return TagResult(
                value=[_entry(d, q.question_id) for d in sorted(dims)],
                source="deterministic", confidence=0.85,
                evidence=ev.hybrid(
                    "question.segment_dimensions.extracted",
                    f"Recognized {len(dims)} standard segmentation dimension(s) in this "
                    "question — from its wording, from its answer options, or both. "
                    "Each is listed below with where it was found.",
                    components=(
                        [ev.component(d, "matched the question title") for d in title_hits]
                        + [ev.component(d, "matched the answer options")
                           for d in sorted(dims - set(title_hits))]
                    ),
                    stage=4,
                    inputs={"dimension_count": len(dims)},
                    quote=q.title,
                ),
            )

        # Segmentable but dimension not inferred — return empty list, not skip
        return TagResult(
            value=[], source="deterministic", confidence=0.60,
            evidence=ev.fallback(
                "question.segment_dimensions.unrecognized",
                "The question IS segmentable, but neither its wording nor its answer "
                "options matched any of the standard dimensions this tagger knows "
                "(department, region, age, tenure, plan...). It probably segments by "
                "something tenant-specific — worth a human look, which is what "
                "separates this empty list from the not-segmentable one above.",
                stage=4,
                inputs={"is_segmentable": "Yes",
                        "option_count": len(q.answer_options)},
            ),
        )


def create_tagger() -> SegmentDimensionsTagger:
    return SegmentDimensionsTagger()
