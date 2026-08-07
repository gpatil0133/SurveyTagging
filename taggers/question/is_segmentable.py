"""is_segmentable tagger — whether responses can segment/cross-tab other data.

Stage 4, deterministic. Depends on role_intent (Stage 3).

Distinction vs is_filterable:
  - filterable = "can be used as a filter facet in UI"
  - segmentable = "can MEANINGFULLY segment OTHER questions' results"
    (demographics, behavioral groups, routing questions)
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


_SEGMENTING_ROLES = {"Segmentation", "Profiling / Demographic"}


class IsSegmentableTagger(QuestionTagger):
    name = "question.is_segmentable"
    tag_dimension = "is_segmentable"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.role_intent"]

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

        role = accumulator.get_question_tag_value(q.question_id, "role_intent")

        # Explicit segmentation roles
        if role in _SEGMENTING_ROLES:
            return TagResult(value="Yes", source="deterministic", confidence=1.0,
                             evidence=f"role_intent={role}")

        # Hidden radio = routing question = segmentable
        if q.question_type == "HR":
            return TagResult(value="Yes", source="deterministic", confidence=0.95,
                             evidence="Hidden radio (routing)")

        # Categorical without weights AND feasible number of buckets (≤15)
        if q.question_type in ("L", "R", "C", "SR"):
            has_weights = any(o.weight is not None for o in q.answer_options)
            n_opts = len(q.answer_options)
            if not has_weights and 2 <= n_opts <= 15:
                return TagResult(value="Yes", source="deterministic", confidence=0.75,
                                 evidence=f"Categorical unweighted with {n_opts} options")

        return TagResult(value="No", source="deterministic", confidence=0.90,
                         evidence=f"Not a segmenting question (type={q.question_type}, role={role})")


def create_tagger() -> IsSegmentableTagger:
    return IsSegmentableTagger()
