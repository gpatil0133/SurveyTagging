"""Trend taggers — whether a question is worth plotting over time, and how
often to bucket it when it is.

Two Stage-4 taggers, deliberately in one module:

    trend_trackable ..... deterministic. The eligibility gate. Depends on
                          metric_type + role_intent (both Stage 3).
    trend_granularity ... hybrid, added in V8. How often to bucket the trend
                          widget, feeding `TrendSetting.FrequencyType` on the
                          payload. Gated on trend_trackable.

They share a module because they share a stage and one depends on the other.
The registry preserves the order `create_tagger()` returns, so the gate always
runs first; splitting them into two files would have sorted `trend_granularity`
ahead of `trend_trackable` alphabetically and read the gate before it was set.

The relationship is a one-way cascade, not a restatement: `trend_trackable`
answers *whether*, `trend_granularity` answers *how often*, and the widget
payload needs both.
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


# Frequency -> the length of one period in days, finest first. The order is the
# search order: start at the cadence's preferred frequency and step coarser
# until the period count is legal.
_FREQUENCIES: tuple[tuple[str, int], ...] = (
    ("Daily", 1),
    ("Weekly", 7),
    ("Monthly", 30),
    ("Quarterly", 91),
    ("Yearly", 365),
)

# The platform's own hard cap: `validate_trend_frequency` rejects any frequency
# that would produce more than 100 periods, so a project spanning three years
# cannot be Weekly however often it actually ships.
_MAX_PERIODS = 100

# Below this many periods a trend line is a couple of dots, so the search steps
# FINER instead — a two-week always-on survey bucketed monthly plots nothing.
_MIN_PERIODS = 3

# project survey_cadence -> where the search starts. How often the survey ships
# is the best available prior for how often it is worth reading.
_CADENCE_PREFERENCE: dict[str, str] = {
    "Always-on": "Weekly",
    "Recurring": "Monthly",
    "Ad-hoc": "Monthly",
}

_DEFAULT_FREQUENCY = "Monthly"


def _resolve_frequency(cadence: str | None, span_days: int) -> tuple[str, int, int]:
    """(frequency, period_count, start_index) for a span of responses.

    Starts from the cadence's preferred frequency, then walks coarser while the
    period count would breach the platform cap and finer while it would be too
    sparse to read. Returns the start index too, so the evidence can say whether
    the answer was the preference or a correction to it.
    """
    names = [f for f, _ in _FREQUENCIES]
    start = names.index(_CADENCE_PREFERENCE.get(cadence or "", _DEFAULT_FREQUENCY))
    index = start

    span = max(int(span_days), 0)
    if span <= 0:
        return names[index], 0, start

    def periods(i: int) -> int:
        return max(1, round(span / _FREQUENCIES[i][1]))

    while index < len(_FREQUENCIES) - 1 and periods(index) > _MAX_PERIODS:
        index += 1
    while index > 0 and periods(index) < _MIN_PERIODS:
        index -= 1

    return names[index], periods(index), start


class TrendGranularityTagger(QuestionTagger):
    """How often to bucket this question's trend widget.

    `trend_trackable` says whether the question belongs on a trend at all; this
    says how often to read it, and the payload's `TrendSetting.FrequencyType`
    needs the second answer as much as the first.

    The frequency is a property of the PROJECT — its cadence and the span its
    responses actually cover — rather than of the question, so every trendable
    question in a survey gets the same answer. It is tagged per question anyway
    because that is where a widget is built from, and because the gate above it
    is per question.
    """

    name = "question.trend_granularity"
    tag_dimension = "trend_granularity"
    stage = 4
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.trend_trackable"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("trend_granularity", stage=4))

        trend = accumulator.get_question_tag_value(q.question_id, "trend_trackable")
        if trend != "Yes":
            return TagResult(
                value="Not Trendable", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.trend_granularity.not_trendable",
                    f"trend_trackable is {trend or 'unset'}, so there is no trend to set "
                    "a frequency for. Not Trendable is a real value rather than a skip: "
                    "a composer reading this dimension alone still gets a usable answer.",
                    stage=4,
                    inputs={"trend_trackable": trend or "(unset)"},
                ),
            )

        cadence = accumulator.get_project_tag_value("survey_cadence")

        # A one-time survey has one wave. Nothing the question supports can make
        # it trendable, so the project verdict outranks the question's own.
        if cadence == "One-time":
            return TagResult(
                value="Not Trendable", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.trend_granularity.one_time_survey",
                    "The question is trend-trackable, but the survey ran once — there is "
                    "no second wave to compare it against. A one-time survey is Not "
                    "Trendable whatever its questions support.",
                    stage=4,
                    inputs={"survey_cadence": "One-time",
                            "trend_trackable": "Yes"},
                ),
            )

        stats = context.response_stats
        span_days = int(stats.span_days) if stats is not None else 0

        if span_days <= 0:
            frequency = _CADENCE_PREFERENCE.get(cadence or "", _DEFAULT_FREQUENCY)
            return TagResult(
                value=frequency, source="hybrid", confidence=0.55,
                evidence=ev.fallback(
                    "question.trend_granularity.cadence_only",
                    f"No response span is available, so the frequency comes from the "
                    f"survey's cadence alone ({cadence or 'unknown'} -> {frequency}). "
                    "That is a prior about how often the survey ships, not a "
                    "measurement of how long it has been collecting — hence 0.55. It "
                    "has NOT been checked against the platform's 100-period cap, "
                    "because there is no span to check.",
                    stage=4,
                    inputs={"survey_cadence": cadence or "(unset)",
                            "span_days": 0},
                ),
            )

        frequency, period_count, start_index = _resolve_frequency(cadence, span_days)
        preferred = [f for f, _ in _FREQUENCIES][start_index]

        if frequency == preferred:
            detail = (
                f"The survey's cadence is {cadence or 'unknown'}, which reads best "
                f"{frequency.lower()}, and {span_days} days of responses gives "
                f"{period_count} period(s) — comfortably inside the platform's "
                f"{_MAX_PERIODS}-period cap."
            )
        else:
            direction = ("coarser" if [f for f, _ in _FREQUENCIES].index(frequency)
                         > start_index else "finer")
            reason = (f"{preferred} over {span_days} days would breach the platform's "
                      f"{_MAX_PERIODS}-period cap, which `validate_trend_frequency` "
                      "rejects outright"
                      if direction == "coarser" else
                      f"{preferred} over only {span_days} days would plot fewer than "
                      f"{_MIN_PERIODS} points, which is a trend line in name only")
            detail = (
                f"The survey's cadence is {cadence or 'unknown'}, which would prefer "
                f"{preferred} — but {reason}. Stepped {direction} to {frequency}, "
                f"giving {period_count} period(s)."
            )

        return TagResult(
            value=frequency, source="hybrid", confidence=0.85,
            evidence=ev.statistic(
                "question.trend_granularity.span_and_cadence",
                detail,
                measure="trend_periods",
                observed=period_count,
                threshold=_MAX_PERIODS,
                stage=4,
                inputs={"survey_cadence": cadence or "(unset)",
                        "span_days": span_days,
                        "preferred_frequency": preferred,
                        "frequency": frequency},
            ),
        )


def create_tagger() -> list[QuestionTagger]:
    # Order matters and is preserved by the registry: the gate runs first, and
    # `trend_granularity` reads its verdict out of the accumulator.
    return [TrendTrackableTagger(), TrendGranularityTagger()]
