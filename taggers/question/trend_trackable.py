"""trend_trackable tagger — whether the question should be tracked over time.

Stage 4, deterministic. Depends on metric_type + role_intent.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class TrendTrackableTagger(QuestionTagger):
    name = "question.trend_trackable"
    tag_dimension = "trend_trackable"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("trend_trackable", stage=4))

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")

        # Standard Metrics always track
        if metric_type == "Standard Metric":
            return TagResult(
                value="Yes", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.trend_trackable.standard_metric",
                    "This is a standard metric (NPS / CSAT / CES / eNPS). Tracking "
                    "these over time is the whole point of running them repeatedly, "
                    "and their definition is stable enough that period-on-period "
                    "comparison is valid.",
                    stage=4,
                    inputs={"metric_type": "Standard Metric"},
                ),
            )

        # Benchmarkable or Primary metric — track
        if role in ("Primary Metric", "Benchmarkable"):
            return TagResult(
                value="Yes", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.trend_trackable.headline_role",
                    f"role_intent is {role} — the question carries a headline or "
                    "benchmarked number, which is exactly what a trend line is for.",
                    stage=4,
                    inputs={"role_intent": role,
                            "metric_type": metric_type or "(unset)"},
                ),
            )

        # Text/verbatim → no trending on raw text
        if metric_type == "Open-ended":
            return TagResult(
                value="No", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.trend_trackable.open_ended",
                    "Raw verbatim text has no value to plot period over period. Themes "
                    "coded out of it can trend, but that happens downstream of tagging.",
                    stage=4,
                    inputs={"metric_type": "Open-ended"},
                ),
            )

        # Segmentation / demographics — don't trend
        if role in ("Segmentation", "Profiling / Demographic", "Screener"):
            return TagResult(
                value="No", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.trend_trackable.grouping_role",
                    f"role_intent is {role}. Questions like this describe WHO answered "
                    "rather than measuring anything — they are the axis you trend other "
                    "metrics along, not a line of their own.",
                    stage=4,
                    inputs={"role_intent": role,
                            "metric_type": metric_type or "(unset)"},
                ),
            )

        # Drivers on known metric — track as supporting
        if role == "Driver / Attribute" and metric_type in ("Standard Metric", "Custom Metric"):
            return TagResult(
                value="Yes", source="deterministic", confidence=0.70,
                evidence=ev.rule(
                    "question.trend_trackable.scored_driver",
                    f"A driver/attribute question sitting on a {metric_type.lower()} "
                    "scale, so it produces a number that can move. Worth trending as a "
                    "supporting line, though it is not a headline metric — hence the "
                    "lower confidence.",
                    stage=4,
                    inputs={"role_intent": "Driver / Attribute",
                            "metric_type": metric_type},
                ),
            )

        return TagResult(
            value="No", source="deterministic", confidence=0.65,
            evidence=ev.fallback(
                "question.trend_trackable.no_rule_matched",
                f"No rule fired: metric_type is {metric_type or 'unset'} and "
                f"role_intent is {role or 'unset'} — neither a standard metric, a "
                "headline role, verbatim text, a grouping question, nor a scored "
                "driver. No is the conservative default here rather than a finding.",
                stage=4,
                inputs={"metric_type": metric_type or "(unset)",
                        "role_intent": role or "(unset)"},
            ),
        )


def create_tagger() -> TrendTrackableTagger:
    return TrendTrackableTagger()
