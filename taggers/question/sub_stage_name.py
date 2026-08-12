"""sub_stage_name tagger — the moment under a journey stage.

Stage 5, LLM-resolved. user_defined, but no longer free text: the value is the
`stages[]` entry of the tenant's own profile journey, selected atomically with
`journey_stage` when the LLM picks a candidate leaf. A tenant whose journey
source has only one level (the EX lifecycle) gets no sub-stage at all rather
than an invented one — under the previous canon path this column defaulted to
"Other <stage>" and filled up with metric names, which is what motivated
sourcing both levels from the profile.

Assigned ONLY to journey-eligible metric questions (NPS/CSAT/CES/custom metric).
Non-eligible questions get status="skipped". Eligible ones are reserved as
pending_llm here and resolved at merge time.
"""

from __future__ import annotations

import logging

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers._metric_utils import is_journey_eligible_metric
from taggers.base import QuestionTagger

logger = logging.getLogger(__name__)


class SubStageNameTagger(QuestionTagger):
    name = "question.sub_stage_name"
    tag_dimension = "sub_stage_name"
    stage = 5
    source_type = "llm"

    @property
    def depends_on(self) -> list[str]:
        return ["question.journey_stage", "question.metric_name"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("sub_stage_name", stage=5))

        # The eligibility check returns its own typed evidence — the "why was
        # this question skipped" answer people actually ask for.
        eligible, evidence = is_journey_eligible_metric(q)
        if not eligible:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=evidence)

        return TagResult(
            value=None, source="llm", confidence=0.0, status="pending_llm",
            evidence=ev.rule(
                "question.sub_stage_name.awaiting_llm",
                "Journey-eligible, so this tagger reserves the dimension and stops. "
                "The value comes from LLM Call 2, which selects a moment from the "
                "tenant's own profile journey ranked by embedding similarity — a "
                "status of pending_llm in the final output means that call never "
                "landed, since a call that lands and declines is recorded as a skip "
                "with its reason. The eligibility reason is recorded on the assigned "
                "tag once it does.",
                stage=5,
            ),
        )


def create_tagger() -> SubStageNameTagger:
    return SubStageNameTagger()
