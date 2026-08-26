"""preferred_segments tagger — which segmenting questions are worth cutting THIS
metric by (V8, Phase 4).

Stage 5, multi-label, llm-refined, user_defined. Depends on is_segmentable
(Stage 4) and metric_type (Stage 3).

**What it fills.** `segmentType` and `segmentationQuestion` on the widget
payload.

**Why it exists.** Segmentation was one-sided before V8. `is_segmentable` and
`segment_dimensions` say a question *can* segment; nothing said which cut is
worth making for a given metric. This is the same reciprocal-pointer pattern as
`driver_link` and `verbatim_link`, applied to the one pairing that was left out
— and the cascade runs one way only, from `is_segmentable` into here, so the two
cannot restate each other.

**Shape.** `[question_id, ...]`, one to three, most relevant first. References
rather than labels, hence `user_defined`.

**Derivation.** The tagger contributes the eligibility gate and the candidate
list; the ranking is LLM Call 2's. That split is deliberate rather than an
economy: which candidates EXIST is a structural fact, and "NPS by region" versus
"NPS by plan tier" is a question about the business, not about the schema. No
rule in this repo can answer the second one, so none pretends to.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# Metric shapes worth segmenting. A segmentation question is not itself
# segmented — that is the other side of the same pairing.
_METRIC_TYPES = {"Standard Metric", "Custom Metric"}

# At most this many are asked for. Past three, a widget carries more breakdowns
# than a reader compares.
MAX_SEGMENTS = 3


def segment_candidates(
    context: UnifiedContext, accumulator: TagAccumulator, exclude_question_id: int
) -> list[QuestionContext]:
    """Every question in the survey that can segment, minus the one being tagged.

    Shared with the prompt builder so the model ranks exactly the candidates
    this tagger declared eligible — the same one-computation-two-consumers shape
    as the journey candidate list.
    """
    out: list[QuestionContext] = []
    for other in context.questions:
        if other.is_content_message or other.question_id == exclude_question_id:
            continue
        if accumulator.get_question_tag_value(other.question_id, "is_segmentable") == "Yes":
            out.append(other)
    return out


class PreferredSegmentsTagger(QuestionTagger):
    name = "question.preferred_segments"
    tag_dimension = "preferred_segments"
    skip_value = []  # multi-label: an empty list, never None
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.is_segmentable", "question.metric_type"]

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        if metric_type not in _METRIC_TYPES:
            return TagResult(
                value=[], source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.preferred_segments.not_a_metric",
                    f"metric_type is {metric_type or 'unset'} — this question is a "
                    "category, a verbatim or an identifier, not something with a number "
                    "to break down. Segmenting it would produce a cross-tab of two "
                    "groupings, which is the other side of this pairing and is what "
                    "is_segmentable records.",
                    stage=5,
                    inputs={"metric_type": metric_type or "(unset)"},
                ),
            )

        candidates = segment_candidates(context, accumulator, q.question_id)
        if not candidates:
            return TagResult(
                value=[], source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.preferred_segments.no_candidates",
                    "This is a metric worth segmenting, but no other question in the "
                    "survey is segmentable — there is no demographic, no routing "
                    "question and no banded metric to cut it by. The empty list is the "
                    "finding: the survey collects a number and nothing to explain it "
                    "with.",
                    stage=5,
                    inputs={"metric_type": metric_type, "candidates": 0},
                ),
            )

        return TagResult(
            value=[], source="hybrid", status="pending_llm", confidence=0.0,
            evidence=ev.rule(
                "question.preferred_segments.awaiting_ranking",
                f"A {metric_type.lower()} with {len(candidates)} segmentable question(s) "
                "in the survey to cut it by. Which of them is worth cutting by is a "
                'question about the business — "NPS by region" versus "NPS by plan '
                'tier" — not about the schema, so the ranking is LLM Call 2\'s and no '
                "rule here guesses at it.",
                stage=5,
                inputs={"metric_type": metric_type,
                        "candidates": len(candidates),
                        "max_segments": MAX_SEGMENTS},
            ),
        )


def create_tagger() -> PreferredSegmentsTagger:
    return PreferredSegmentsTagger()
