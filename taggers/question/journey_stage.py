"""journey_stage tagger — V5: pending_llm-only.

Stage 5 LLM tagger. Eligibility (NPS/CSAT/CES/custom metric) is enforced
locally; the actual stage assignment comes from the canon-aware LLM call
at orchestrator merge time. Keyword priors and the NPS->advocacy heuristic
are gone because they (a) operate in the YAML namespace which now diverges
from the tenant canon and (b) misfired badly on generic words like
"experience" and "help" (see TENANT_PROFILE_PLAN forensic).

Non-eligible questions get status="skipped". Eligible questions get
status="pending_llm" — the merge step writes "assigned" or
"low_confidence_assigned" depending on LLM output.
"""

from __future__ import annotations

import logging

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers._metric_utils import is_journey_eligible_metric
from taggers.base import QuestionTagger

logger = logging.getLogger(__name__)


class JourneyStageTagger(QuestionTagger):
    name = "question.journey_stage"
    tag_dimension = "journey_stage"
    stage = 5
    source_type = "llm"

    def __init__(self, registry=None) -> None:
        # `registry` accepted for back-compat with existing test fixtures and
        # callers that previously passed an `IndustryStagesRegistry`. V5
        # journey_stage assignment is canon-driven through the LLM merge
        # step; the registry is no longer used here.
        super().__init__()
        del registry  # explicitly unused

    @property
    def depends_on(self) -> list[str]:
        return ["question.role_intent", "question.metric_name", "question.metric_type"]

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

        eligible, evidence = is_journey_eligible_metric(q)
        if not eligible:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=evidence)

        return TagResult(value=None, source="llm", confidence=0.0, status="pending_llm",
                         evidence="Requires LLM classification")


def create_tagger() -> JourneyStageTagger:
    return JourneyStageTagger()
