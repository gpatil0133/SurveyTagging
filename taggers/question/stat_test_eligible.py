"""stat_test_eligible tagger — what statistical tests can be applied to this question.

Stage 4, deterministic. Depends on metric_type (Stage 3).
"""

from __future__ import annotations

from models import evidence as ev
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
                             evidence=ev.content_message("stat_test_eligible", stage=4))

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        has_weights = any(o.weight is not None for o in q.answer_options)
        n_opts = len(q.answer_options)

        # Text — cannot run parametric tests
        if metric_type == "Open-ended":
            return TagResult(
                value="No", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.stat_test.open_ended",
                    "The answers are free text. There is no numeric or categorical "
                    "variable to test — significance testing would have to run on "
                    "coded themes, which happens downstream of tagging.",
                    stage=4,
                    inputs={"metric_type": "Open-ended"},
                ),
            )

        # Standard/Custom metric with numeric weights — parametric
        if metric_type in ("Standard Metric", "Custom Metric") and has_weights and n_opts >= 2:
            return TagResult(
                value="Yes - Parametric", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.stat_test.weighted_scale",
                    f"A {metric_type.lower()} whose {n_opts} options carry numeric "
                    "weights. Weighted options make the responses interval-scaled, so "
                    "means are meaningful and t-tests / ANOVA apply.",
                    stage=4,
                    inputs={"metric_type": metric_type,
                            "option_count": n_opts,
                            "has_weighted_options": True},
                ),
            )

        # Categorical with 2+ options — chi-square
        if metric_type == "Categorical" and n_opts >= 2:
            return TagResult(
                value="Yes - Chi-square", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.stat_test.categorical",
                    f"Categorical answers across {n_opts} options. There is no "
                    "meaningful mean to compare, so the test is one of independence "
                    "between counts — chi-square.",
                    stage=4,
                    inputs={"metric_type": "Categorical", "option_count": n_opts},
                ),
            )

        # Rating scales without weights — non-parametric (Mann-Whitney / Kruskal-Wallis)
        if metric_type in ("Standard Metric", "Custom Metric") and n_opts >= 2:
            return TagResult(
                value="Yes - Non-parametric", source="deterministic", confidence=0.75,
                evidence=ev.rule(
                    "question.stat_test.unweighted_scale",
                    f"A {metric_type.lower()} with {n_opts} options but no weights on "
                    "them. The responses are ordered but the spacing between them is "
                    "unknown, so rank-based tests (Mann-Whitney, Kruskal-Wallis) are "
                    "the honest choice rather than a t-test.",
                    stage=4,
                    inputs={"metric_type": metric_type,
                            "option_count": n_opts,
                            "has_weighted_options": False},
                ),
            )

        return TagResult(
            value="No", source="deterministic", confidence=0.80,
            evidence=ev.rule(
                "question.stat_test.insufficient_structure",
                f"metric_type is {metric_type or 'unset'} with {n_opts} answer "
                "option(s) — too few options, or no measurement structure at all, to "
                "compare groups against.",
                stage=4,
                inputs={"metric_type": metric_type or "(unset)",
                        "option_count": n_opts,
                        "has_weighted_options": has_weights},
            ),
        )


def create_tagger() -> StatTestEligibleTagger:
    return StatTestEligibleTagger()
