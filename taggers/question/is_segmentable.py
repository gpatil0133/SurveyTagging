"""is_segmentable tagger — whether responses can segment/cross-tab other data.

Stage 4, deterministic. Depends on role_intent (Stage 3).

Distinction vs is_filterable:
  - filterable = "can be used as a filter facet in UI"
  - segmentable = "can MEANINGFULLY segment OTHER questions' results"
    (demographics, behavioral groups, routing questions)
"""

from __future__ import annotations

from models import evidence as ev
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
                             evidence=ev.content_message("is_segmentable", stage=4))

        role = accumulator.get_question_tag_value(q.question_id, "role_intent")

        # Explicit segmentation roles
        if role in _SEGMENTING_ROLES:
            return TagResult(
                value="Yes", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.is_segmentable.segmenting_role",
                    f"role_intent is already {role}, which exists precisely to slice "
                    "other questions' results by. Note this dimension is about "
                    "segmenting OTHER questions, not about being filterable itself.",
                    stage=4,
                    inputs={"role_intent": role},
                ),
            )

        # Hidden radio = routing question = segmentable
        if q.question_type == "HR":
            return TagResult(
                value="Yes", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.is_segmentable.hidden_radio",
                    "A hidden radio (type HR) is set by the survey logic rather than "
                    "the respondent — it records which branch someone took, which is a "
                    "natural grouping for comparing everyone else's answers.",
                    stage=4,
                    inputs={"question_type": "HR", "role_intent": role or "(unset)"},
                ),
            )

        # Categorical without weights AND feasible number of buckets (≤15)
        if q.question_type in ("L", "R", "C", "SR"):
            has_weights = any(o.weight is not None for o in q.answer_options)
            n_opts = len(q.answer_options)
            if not has_weights and 2 <= n_opts <= 15:
                return TagResult(
                    value="Yes", source="deterministic", confidence=0.75,
                    evidence=ev.statistic(
                        "question.is_segmentable.usable_buckets",
                        f"An unweighted categorical question with {n_opts} options: "
                        "enough groups to compare, and few enough that each keeps a "
                        "usable sample size. Past 15 options the cells get too thin to "
                        "read.",
                        measure="option_count",
                        observed=n_opts,
                        threshold=15,
                        stage=4,
                        inputs={"question_type": q.question_type,
                                "role_intent": role or "(unset)"},
                    ),
                )

        return TagResult(
            value="No", source="deterministic", confidence=0.90,
            evidence=ev.rule(
                "question.is_segmentable.not_a_grouping",
                f"Nothing makes this a grouping variable: its role is "
                f"{role or 'unset'} rather than segmentation or demographic, it is not "
                f"a routing question, and type {q.question_type} with "
                f"{len(q.answer_options)} option(s) does not give clean unweighted "
                "buckets.",
                stage=4,
                inputs={"question_type": q.question_type,
                        "role_intent": role or "(unset)",
                        "option_count": len(q.answer_options)},
            ),
        )


def create_tagger() -> IsSegmentableTagger:
    return IsSegmentableTagger()
