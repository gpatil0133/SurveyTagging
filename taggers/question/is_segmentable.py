"""is_segmentable tagger — whether responses can segment/cross-tab other data.

Stage 4, deterministic. Depends on role_intent (Stage 3).

Since V7.3 this is the *only* statement of the capability: `control_role`'s
`Segment-Control` was removed rather than kept in sync, because it was derived at
Stage 3 and so could not see `role_intent` — the first and strongest rule here.
A dashboard's segment picker reads this dimension.

Distinction vs is_filterable:
  - filterable = "can be used as a filter facet in UI"
  - segmentable = "can MEANINGFULLY segment OTHER questions' results"
    (demographics, behavioral groups, routing questions, metric bands)

Metric bands qualify: "everything else by NPS band" is the standard driver read,
and it is the same comparison as "NPS by region" seen from the other side. Only
PLATFORM-scored metrics get it — they are the ones with canonical bands. An
unflagged rating scale is filterable (a scale point is a facet) without being
segmentable (no agreed banding to group by), which is exactly the difference
this dimension exists to record.

V8 re-verified this against the three response formats the Phase 1 split added
(`Date`, `Numeric-Open`, `File-Upload`). All three are `T` questions, which is
not in `_BUCKETING_TYPES`, so all three resolve to No unless the platform has
flagged them as a metric — and none of them can be, since a text question
carries no rs_type. Nothing to group other answers by: a date is continuous, a
number is unbounded, an attachment is not an answer. No rule changed.

Since V8 this dimension also gates `preferred_segments`, which asks the
reciprocal question — not "can this question segment?" but "which segmenting
question is worth cutting THIS metric by?". That is a one-way cascade: the
candidate list for `preferred_segments` is every question this dimension says
Yes to.
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


_SEGMENTING_ROLES = {"Segmentation", "Profiling / Demographic"}

# Choice types whose options form groups you can break other questions out by.
# Mirrors `_FILTERABLE_TYPES` in is_filterable.py minus HR, which is handled by
# its own branch above (a routing question qualifies whatever its option count).
_BUCKETING_TYPES = {"L", "R", "C", "SR", "ML"}


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

        # Platform-scored metric = canonical bands = a grouping variable.
        if is_platform_metric(q):
            return TagResult(
                value="Yes", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.is_segmentable.platform_metric",
                    f"This is {describe_scale_kind(bounded_scale_kind(q) or '')}. Those "
                    "bands are a grouping variable, not just a filter: comparing every "
                    "other answer across promoters and detractors is the standard "
                    "driver read. Scales the platform has NOT flagged fall through to "
                    "the bucket rule below — they have no agreed banding to group by.",
                    stage=4,
                    inputs={"rs_type": q.rs_type,
                            "is_custom_metric": q.is_custom_metric,
                            "role_intent": role or "(unset)"},
                ),
            )

        # Categorical without weights AND feasible number of buckets (≤15).
        # ML (multi-select) belongs here: a respondent picking several options
        # lands in several groups, which is how "by product owned" is read. It
        # was missing until V7.3, and the omission was invisible because the
        # since-removed `control_role` computed the same capability from
        # `response_format` (where ML *is* a select shape) and disagreed.
        if q.question_type in _BUCKETING_TYPES:
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
                f"a routing question, the platform does not score it as a metric (so "
                f"there are no bands to group by), and type {q.question_type} with "
                f"{len(q.answer_options)} option(s) does not give clean unweighted "
                "buckets. It may still be filterable — see is_filterable.",
                stage=4,
                inputs={"question_type": q.question_type,
                        "rs_type": q.rs_type,
                        "role_intent": role or "(unset)",
                        "option_count": len(q.answer_options)},
            ),
        )


def create_tagger() -> IsSegmentableTagger:
    return IsSegmentableTagger()
