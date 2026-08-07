"""stat_test_eligible tagger — what statistical tests can be applied to this question.

Stage 4, deterministic. Depends on metric_type (Stage 3).
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class StatTestEligibleTagger(QuestionTagger):
    name = "question.stat_test_eligible"
    tag_dimension = "stat_test_eligible"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type"]

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

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        has_weights = any(o.weight is not None for o in q.answer_options)
        n_opts = len(q.answer_options)

        # Text — cannot run parametric tests
        if metric_type == "Open-ended":
            return TagResult(value="No", source="deterministic", confidence=1.0,
                             evidence="Open-ended text — no statistical test applicable")

        # Standard/Custom metric with numeric weights — parametric
        if metric_type in ("Standard Metric", "Custom Metric") and has_weights and n_opts >= 2:
            return TagResult(value="Yes - Parametric", source="deterministic",
                             confidence=0.95,
                             evidence=f"Rating scale with weights ({n_opts} options)")

        # Categorical with 2+ options — chi-square
        if metric_type == "Categorical" and n_opts >= 2:
            return TagResult(value="Yes - Chi-square", source="deterministic",
                             confidence=0.90,
                             evidence=f"Categorical with {n_opts} options (chi-square)")

        # Rating scales without weights — non-parametric (Mann-Whitney / Kruskal-Wallis)
        if metric_type in ("Standard Metric", "Custom Metric") and n_opts >= 2:
            return TagResult(value="Yes - Non-parametric", source="deterministic",
                             confidence=0.75,
                             evidence="Rating without weights — rank-based tests")

        return TagResult(value="No", source="deterministic", confidence=0.80,
                         evidence=f"Insufficient variance or structure (type={metric_type}, opts={n_opts})")


def create_tagger() -> StatTestEligibleTagger:
    return StatTestEligibleTagger()
