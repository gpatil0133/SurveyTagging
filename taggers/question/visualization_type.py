"""visualization_type tagger — recommended chart type.

Stage 5, hybrid. Depends on metric_type, role_intent (Stage 3) + trend_trackable (Stage 4).
Per F1: no same-stage dependencies.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class VisualizationTypeTagger(QuestionTagger):
    name = "question.visualization_type"
    tag_dimension = "visualization_type"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent", "question.trend_trackable"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("visualization_type", stage=5))

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")
        trend = accumulator.get_question_tag_value(q.question_id, "trend_trackable")

        # Open-ended → Word Cloud
        if metric_type == "Open-ended":
            return TagResult(
                value="Word Cloud", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.visualization_type.open_ended",
                    "The answers are free text, so there is no axis to plot. A word "
                    "cloud is the only summary available before the verbatims are "
                    "theme-coded.",
                    stage=5,
                    inputs={"metric_type": "Open-ended"},
                ),
            )

        # Primary Metric with Standard Metric → Score Card
        if role == "Primary Metric" and metric_type == "Standard Metric":
            # If trend-trackable + we had history → would use Line Chart; per F7,
            # we emit Score Card for per-survey projection and let needs_history flag
            # upgrade in Phase 4 aggregation.
            return TagResult(
                value="Score Card", source="deterministic", confidence=0.95,
                evidence=ev.hybrid(
                    "question.visualization_type.headline_scorecard",
                    "A standard metric in the survey's primary role — the one number "
                    "someone opens the dashboard to see. A score card shows it big "
                    "rather than burying it in a chart. (A line chart would be better "
                    "once history exists; that upgrade happens at aggregation time, "
                    "not here.)",
                    components=[
                        ev.component("role_intent", "Primary Metric"),
                        ev.component("metric_type", "Standard Metric"),
                    ],
                    stage=5,
                    inputs={"trend_trackable": trend or "(unset)"},
                ),
            )

        # Driver in matrix → Heat Map
        if role == "Driver / Attribute" and q.matrix_group_title:
            return TagResult(
                value="Heat Map", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.visualization_type.matrix_driver",
                    "A driver question inside a matrix. Its siblings share a scale and "
                    "are read together, and a heat map is what makes the strong and "
                    "weak rows visible at a glance.",
                    stage=5,
                    inputs={"role_intent": "Driver / Attribute",
                            "matrix_group_title": q.matrix_group_title},
                ),
            )

        # Categorical
        if metric_type == "Categorical":
            n_opts = len(q.answer_options)
            if n_opts == 2:
                return TagResult(
                    value="Pie Chart", source="deterministic", confidence=0.80,
                    evidence=ev.rule(
                        "question.visualization_type.binary_pie",
                        "Two mutually exclusive options that sum to the whole — the "
                        "one case where a pie chart genuinely reads better than bars.",
                        stage=5,
                        inputs={"metric_type": "Categorical", "option_count": 2},
                    ),
                )
            return TagResult(
                value="Bar Chart", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.visualization_type.categorical_bars",
                    f"{n_opts} categories to compare. Bars put them on a common "
                    "baseline, which is what makes the comparison readable.",
                    stage=5,
                    inputs={"metric_type": "Categorical", "option_count": n_opts},
                ),
            )

        # Rating scale default
        if metric_type in ("Standard Metric", "Custom Metric"):
            return TagResult(
                value="Bar Chart", source="deterministic", confidence=0.75,
                evidence=ev.rule(
                    "question.visualization_type.scale_default",
                    f"A {metric_type.lower()} that is neither the survey's headline "
                    "number nor a matrix driver. Bars over the scale points show the "
                    "distribution, not just the average.",
                    stage=5,
                    inputs={"metric_type": metric_type,
                            "role_intent": role or "(unset)"},
                ),
            )

        # Fallback (LLM refines)
        return TagResult(
            value="Table", source="hybrid", confidence=0.40,
            evidence=ev.fallback(
                "question.visualization_type.no_rule_matched",
                f"metric_type is {metric_type or 'unset'} — not text, categorical or a "
                "scale — so no chart rule applies. A table is the honest fallback for "
                "data whose shape is unknown, and the 0.40 invites LLM Call 2 to do "
                "better.",
                stage=5,
                inputs={"metric_type": metric_type or "(unset)",
                        "role_intent": role or "(unset)",
                        "question_type": q.question_type},
            ),
        )


def create_tagger() -> VisualizationTypeTagger:
    return VisualizationTypeTagger()
