"""is_filterable tagger — whether the question works as a filter/facet in reports.

Stage 3, deterministic. No tag dependencies.

Two things make a question filterable, not one: a bounded set of answer choices
(a choice list) *or* a bounded scale (a metric). Metrics were missing — a survey
whose only NPS item reported `is_filterable: No` was telling consumers they
could not build a "Promoters only" facet, which every reporting UI offers. The
scale predicate lives in `taggers/_metric_utils.py`, shared with the
dashboard-capability layer so the two cannot resolve it differently.

Since V7.3 this is the *only* statement of the capability: `control_role`'s
`Dropdown-Filter` was removed rather than kept in sync with it. A dashboard's
filter bar reads this dimension; free-text search reads `response_format ==
"Open-Text"`, which is the one control this boolean deliberately says No to (an
unbounded answer builds no facet, even though it can be searched).
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers._metric_utils import (
    bounded_scale_kind,
    describe_scale_kind,
    is_platform_metric,
)
from taggers.base import QuestionTagger


_FILTERABLE_TYPES = {"L", "R", "C", "HR", "ML", "SR"}


class IsFilterableTagger(QuestionTagger):
    name = "question.is_filterable"
    tag_dimension = "is_filterable"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("is_filterable", stage=3))

        # Metrics first: the platform's scale flags outrank the question type, so
        # a CSAT delivered as a radio button is filtered on as a CSAT.
        kind = bounded_scale_kind(q)
        if kind:
            platform = is_platform_metric(q)
            return TagResult(
                value="Yes", source="deterministic",
                # Read straight off a platform flag vs inferred from the question
                # type, same split as metric_type.
                confidence=1.0 if platform else 0.90,
                evidence=ev.rule(
                    "question.is_filterable.bounded_scale",
                    f"This is {describe_scale_kind(kind)}. Answers land on a bounded, "
                    "discrete scale, so the reporting UI can facet on a score band or "
                    "a single scale point — a metric is something you can filter BY, "
                    "not only something you filter.",
                    stage=3,
                    inputs={"question_type": q.question_type,
                            "rs_type": q.rs_type,
                            "is_custom_metric": q.is_custom_metric,
                            "scale_kind": kind},
                ),
            )

        if q.question_type in _FILTERABLE_TYPES:
            return TagResult(
                value="Yes", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.is_filterable.categorical_type",
                    f"Question type {q.question_type} produces a bounded set of discrete "
                    "answer choices, which is exactly what a report filter needs to "
                    "build a facet from.",
                    stage=3,
                    inputs={"question_type": q.question_type},
                ),
            )

        # Explicit No for text/contact/signature and anything unrecognized.
        return TagResult(
            value="No", source="deterministic", confidence=1.0,
            evidence=ev.rule(
                "question.is_filterable.unbounded_answer",
                f"Question type {q.question_type} is free text, a contact block or a "
                "signature — the answer is neither one of a fixed list of choices nor a "
                "point on a scale, so there is no bounded set to build a facet from. "
                "(Metrics and rating scales are checked first and do qualify.)",
                stage=3,
                inputs={"question_type": q.question_type,
                        "rs_type": q.rs_type,
                        "filterable_types": sorted(_FILTERABLE_TYPES)},
            ),
        )


def create_tagger() -> IsFilterableTagger:
    return IsFilterableTagger()
