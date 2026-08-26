"""display_role tagger — how the question is displayed on dashboards.

Stage 5, hybrid. Depends on metric_type, role_intent (Stage 3), trend_trackable (Stage 4).

display_role depends on dashboard_placement (both Stage 5). Per F1: alphabetical
"da" < "di", so dashboard_placement runs BEFORE display_role within Stage 5. ✓
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class DisplayRoleTagger(QuestionTagger):
    name = "question.display_role"
    tag_dimension = "display_role"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent", "question.trend_trackable",
                "question.dashboard_placement"]

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")
        trend = accumulator.get_question_tag_value(q.question_id, "trend_trackable")

        if role == "Primary Metric" and metric_type == "Standard Metric":
            return TagResult(
                value="Primary KPI", source="deterministic", confidence=0.95,
                evidence=ev.hybrid(
                    "question.display_role.standard_primary",
                    "A standard metric in the primary role — a benchmarkable headline "
                    "number. This is what goes at the top of the dashboard.",
                    components=[
                        ev.component("role_intent", "Primary Metric"),
                        ev.component("metric_type", "Standard Metric"),
                    ],
                    stage=5,
                ),
            )

        if role == "Benchmarkable":
            return TagResult(
                value="Comparison Metric", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.display_role.benchmarkable",
                    "role_intent is Benchmarkable — the question exists to be compared "
                    "against an external or historical reference, so it is displayed "
                    "next to that reference rather than on its own.",
                    stage=5,
                    inputs={"role_intent": "Benchmarkable"},
                ),
            )

        if role == "Driver / Attribute":
            return TagResult(
                value="Supporting Metric", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.display_role.driver",
                    "role_intent is Driver / Attribute — it explains why the headline "
                    "number moved, so it belongs beneath that number rather than "
                    "competing with it.",
                    stage=5,
                    inputs={"role_intent": "Driver / Attribute"},
                ),
            )

        if role == "Follow-up / Verbatim":
            return TagResult(
                value="Detail View", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.display_role.verbatim",
                    "role_intent is Follow-up / Verbatim. Free-text answers do not "
                    "summarize into a dashboard tile — they are read individually, on "
                    "a detail view.",
                    stage=5,
                    inputs={"role_intent": "Follow-up / Verbatim"},
                ),
            )

        if role == "Diagnostic":
            return TagResult(
                value="Drill-down", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.display_role.diagnostic",
                    "role_intent is Diagnostic — the question is asked to explain a "
                    "specific problem, which is something a reader goes looking for "
                    "rather than something shown up front.",
                    stage=5,
                    inputs={"role_intent": "Diagnostic"},
                ),
            )

        if role == "Primary Metric":
            return TagResult(
                value="Primary KPI", source="deterministic", confidence=0.80,
                evidence=ev.rule(
                    "question.display_role.custom_primary",
                    f"role_intent is Primary Metric, so this is the survey's headline "
                    f"number — but metric_type is {metric_type or 'unset'} rather than "
                    "a standard metric, so it cannot be benchmarked outside this "
                    "tenant. Hence 0.80 rather than 0.95.",
                    stage=5,
                    inputs={"role_intent": "Primary Metric",
                            "metric_type": metric_type or "(unset)"},
                ),
            )

        if trend == "Yes":
            return TagResult(
                value="Trend Indicator", source="hybrid", confidence=0.65,
                evidence=ev.rule(
                    "question.display_role.trendable",
                    f"No role rule matched (role_intent is {role or 'unset'}), but "
                    "trend_trackable is Yes — the question produces a number that can "
                    "move, so it is displayed as a movement rather than a level.",
                    stage=5,
                    inputs={"trend_trackable": "Yes",
                            "role_intent": role or "(unset)"},
                ),
            )

        return TagResult(
            value="Detail View", source="hybrid", confidence=0.45,
            evidence=ev.fallback(
                "question.display_role.no_rule_matched",
                f"No rule fired: role_intent is {role or 'unset'}, metric_type is "
                f"{metric_type or 'unset'}, and the question does not trend. Detail "
                "View is the parking space for questions with no clear dashboard job — "
                "the 0.45 is there so LLM Call 2 overrides it.",
                stage=5,
                inputs={"role_intent": role or "(unset)",
                        "metric_type": metric_type or "(unset)",
                        "trend_trackable": trend or "(unset)"},
            ),
        )


def create_tagger() -> DisplayRoleTagger:
    return DisplayRoleTagger()
