"""calculation_type tagger — how the question's responses should be aggregated.

Stage 4, deterministic. Depends on metric_type + metric_name (both Stage 3).
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class CalculationTypeTagger(QuestionTagger):
    name = "question.calculation_type"
    tag_dimension = "calculation_type"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.metric_name"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("calculation_type", stage=4))

        metric_name = accumulator.get_question_tag_value(q.question_id, "metric_name")
        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")

        # Platform source hint — trust if non-empty
        if q.calculation_type:
            # Normalize common Sogolytics names to our enum
            ct = q.calculation_type.strip().lower()
            def _from_platform(value: str) -> TagResult:
                return TagResult(
                    value=value, source="deterministic", confidence=1.0,
                    evidence=ev.rule(
                        "question.calculation_type.platform_declared",
                        f"The survey author configured this question's calculation in "
                        f"the platform itself, and that setting normalizes to {value}. "
                        "An explicit author choice outranks every inference below.",
                        stage=4,
                        inputs={"platform_calculation_type": q.calculation_type,
                                "normalized_to": value},
                    ),
                )

            if "weighted" in ct or "mean" in ct or "average" in ct:
                return _from_platform("Mean")
            if "percentage" in ct or "percent" in ct or ct == "%":
                return _from_platform("Percentage")
            if "nps" in ct:
                return _from_platform("NPS Score")
            if "sum" in ct:
                return _from_platform("Sum")

        # Named standard metrics
        if metric_name in ("NPS", "eNPS"):
            return TagResult(
                value="NPS Score", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.calculation_type.nps_definition",
                    f"{metric_name} has one correct aggregation by definition — "
                    "%promoters minus %detractors — not a mean of the 0-10 scores.",
                    stage=4,
                    inputs={"metric_name": metric_name},
                ),
            )
        if metric_name in ("CSAT", "CES"):
            return TagResult(
                value="Mean", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.calculation_type.scored_standard_metric",
                    f"{metric_name} is reported as an average of the rating scale. "
                    "(Some organisations prefer top-box percentage instead; this "
                    "pipeline reports the mean, which is why the confidence is not 1.0.)",
                    stage=4,
                    inputs={"metric_name": metric_name},
                ),
            )

        # Ranking
        if q.question_type in ("RW", "RK"):
            return TagResult(
                value="Mean Rank", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.calculation_type.ranking",
                    f"Type {q.question_type} asks respondents to order options. The "
                    "answer is a position, so the aggregate is an average rank rather "
                    "than an average value.",
                    stage=4,
                    inputs={"question_type": q.question_type},
                ),
            )

        # By metric_type + options count
        if metric_type == "Open-ended":
            return TagResult(
                value="Count", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.calculation_type.open_ended",
                    "Free-text answers cannot be averaged or summed. The only "
                    "aggregate available before theme coding is how many people "
                    "responded.",
                    stage=4,
                    inputs={"metric_type": "Open-ended"},
                ),
            )

        if metric_type == "Categorical":
            n_opts = len(q.answer_options)
            if n_opts == 2:
                return TagResult(
                    value="Percentage", source="deterministic", confidence=0.90,
                    evidence=ev.rule(
                        "question.calculation_type.binary_choice",
                        "A two-option categorical question. With only two outcomes, "
                        "one percentage tells the whole story — the other is its "
                        "complement.",
                        stage=4,
                        inputs={"metric_type": "Categorical", "option_count": 2},
                    ),
                )
            if n_opts >= 3:
                return TagResult(
                    value="Distribution", source="deterministic", confidence=0.85,
                    evidence=ev.rule(
                        "question.calculation_type.multi_option",
                        f"A categorical question with {n_opts} options. No single "
                        "number represents it — the shape across all options is the "
                        "result.",
                        stage=4,
                        inputs={"metric_type": "Categorical",
                                "option_count": n_opts},
                    ),
                )

        if metric_type in ("Standard Metric", "Custom Metric"):
            # Rating scales default to Mean
            return TagResult(
                value="Mean", source="deterministic", confidence=0.80,
                evidence=ev.rule(
                    "question.calculation_type.scale_default",
                    f"A {metric_type.lower()} that the platform did not configure and "
                    "that is not one of the named standard metrics. Rating scales "
                    "default to a mean; the 0.80 flags that this is the scale default "
                    "rather than a declared choice.",
                    stage=4,
                    inputs={"metric_type": metric_type,
                            "metric_name": metric_name or "(unset)"},
                ),
            )

        # Not applicable
        return TagResult(
            value="Not Applicable", source="deterministic", confidence=0.70,
            evidence=ev.fallback(
                "question.calculation_type.no_rule_matched",
                f"Nothing to aggregate: type {q.question_type} with metric_type "
                f"{metric_type or 'unset'} matched no platform setting, named metric, "
                "ranking, text, categorical or scale rule. Typically a contact block "
                "or an unrecognized question type.",
                stage=4,
                inputs={"question_type": q.question_type,
                        "metric_type": metric_type or "(unset)",
                        "metric_name": metric_name or "(unset)"},
            ),
        )


def create_tagger() -> CalculationTypeTagger:
    return CalculationTypeTagger()
