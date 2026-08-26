"""visualization_type tagger — recommended chart type.

Stage 5, hybrid. Depends on response_format, metric_type, role_intent (Stage 3)
+ trend_trackable (Stage 4). Per F1: no same-stage dependencies.

V8 moved the vocabulary onto the platform's own chart names, so this dimension
and `widget_compatibility` speak one chart language and a composer never has to
translate between two. Orientation is a real distinction here — horizontal and
vertical bars are separate codes — so `Bar Chart` split, `Score Card` took the
platform's name (`Number`), and three values with no chart code behind them were
dropped: `Heat Map` (a grid may only use the stacked variants, so its default
became `Horizontal Stacked Bar`), `Distribution Chart` (a bar chart on a Group
Percentage metric expresses it) and `Trend Line` (a display mode plus Line or
Area, not a chart — its information moved to `trend_granularity`).

Nothing was ADDED beyond the rename targets. A default should stay conservative;
the exotic charts belong in `widget_compatibility`'s allow-list, where the
dashboard service can reach for one deliberately.

The rule table also branches on `response_format` first now. It used to key off
`metric_type`, where every text question is `Open-ended` — so a date picker, a
numeric field and a file upload all defaulted to a word cloud.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# response_format -> the one default chart, for the shapes where the format
# alone decides. Checked before the role/metric_type rules below, because these
# formats have either exactly one legal chart or one obviously right one.
_FORMAT_DEFAULTS: dict[str, tuple[str, str]] = {
    "Date": ("Line Chart",
             "Count is the only metric the platform allows on a date question, and a "
             "count per period is a line — the shape of when responses happened is the "
             "whole content of the answer."),
    "Numeric-Open": ("Number",
                     "A statistical text field answers with a real number. The headline "
                     "is the aggregate itself; the distribution behind it is available "
                     "in widget_compatibility when someone wants it."),
    "Contact": ("Table",
                "Identity capture. There is nothing to plot; a table is the only honest "
                "rendering."),
}


class VisualizationTypeTagger(QuestionTagger):
    name = "question.visualization_type"
    tag_dimension = "visualization_type"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.response_format", "question.metric_type",
                "question.role_intent", "question.trend_trackable"]

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        fmt = accumulator.get_question_tag_value(q.question_id, "response_format")
        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")
        trend = accumulator.get_question_tag_value(q.question_id, "trend_trackable")

        # A file upload has no analyzable answer at all — widget_compatibility
        # is empty for it, so naming a default chart here would contradict the
        # dimension that says nothing can render this question. Skipped rather
        # than fallen back to Table. (Before V8 it became a word cloud of
        # filenames, which is where the contradiction was invisible.)
        if fmt == "File-Upload":
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.visualization_type.nothing_to_render",
                    "A file upload has no analyzable answer — only attachments. "
                    "widget_compatibility is empty for this shape, so there is no "
                    "chart to recommend and naming one would contradict it.",
                    stage=5,
                    inputs={"response_format": fmt,
                            "question_sub_type": q.question_sub_type},
                ),
            )

        # Format-decided shapes first. These used to fall into the Open-ended
        # branch below and come out as word clouds.
        if fmt in _FORMAT_DEFAULTS:
            value, why = _FORMAT_DEFAULTS[fmt]
            return TagResult(
                value=value, source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.visualization_type.format_default",
                    why,
                    stage=5,
                    inputs={"response_format": fmt,
                            "question_sub_type": q.question_sub_type},
                ),
            )

        # Genuine free text → Word Cloud
        if fmt == "Open-Text" or (fmt is None and metric_type == "Open-ended"):
            return TagResult(
                value="Word Cloud", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.visualization_type.open_ended",
                    "The answers are free text, so there is no axis to plot. A word "
                    "cloud is the only summary available before the verbatims are "
                    "theme-coded. (Note widget_compatibility lists Table alone here: "
                    "the word-cloud widget type has no creator on the platform yet, so "
                    "this is the right default and not yet a buildable one.)",
                    stage=5,
                    inputs={"response_format": fmt or "(unset)",
                            "metric_type": metric_type or "(unset)"},
                ),
            )

        # Primary Metric with Standard Metric → Number
        if role == "Primary Metric" and metric_type == "Standard Metric":
            # If trend-trackable + we had history → would use Line Chart; per F7,
            # we emit the headline number for per-survey projection and let
            # needs_history flag upgrade in Phase 4 aggregation.
            return TagResult(
                value="Number", source="deterministic", confidence=0.95,
                evidence=ev.hybrid(
                    "question.visualization_type.headline_number",
                    "A standard metric in the survey's primary role — the one number "
                    "someone opens the dashboard to see. The platform's Number widget "
                    "shows it big rather than burying it in a chart. (A line chart would "
                    "be better once history exists; that upgrade happens at aggregation "
                    "time, not here.)",
                    components=[
                        ev.component("role_intent", "Primary Metric"),
                        ev.component("metric_type", "Standard Metric"),
                    ],
                    stage=5,
                    inputs={"trend_trackable": trend or "(unset)"},
                ),
            )

        # Matrix row → the stacked variants are the only legal charts
        if fmt == "Matrix-Row" or (role == "Driver / Attribute" and q.matrix_group_title):
            return TagResult(
                value="Horizontal Stacked Bar", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.visualization_type.matrix_row",
                    "A row of a matrix. Its siblings share a scale and are read "
                    "together, and the platform permits exactly two charts on a grid "
                    "question — the horizontal and vertical stacked bar. Horizontal is "
                    "the readable default when the row labels are sentences. (This used "
                    "to be Heat Map, which has no chart code and could never be built.)",
                    stage=5,
                    inputs={"response_format": fmt or "(unset)",
                            "role_intent": role or "(unset)",
                            "matrix_group_title": q.matrix_group_title or "(none)"},
                ),
            )

        # Categorical
        if metric_type == "Categorical":
            n_opts = len(q.answer_options)
            if n_opts == 2 and fmt != "Multi-Select":
                return TagResult(
                    value="Pie Chart", source="deterministic", confidence=0.80,
                    evidence=ev.rule(
                        "question.visualization_type.binary_pie",
                        "Two mutually exclusive options that sum to the whole — the "
                        "one case where a pie chart genuinely reads better than bars.",
                        stage=5,
                        inputs={"metric_type": "Categorical", "option_count": 2,
                                "response_format": fmt or "(unset)"},
                    ),
                )
            return TagResult(
                value="Horizontal Bar", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.visualization_type.categorical_bars",
                    f"{n_opts} categories to compare. Bars put them on a common "
                    "baseline, which is what makes the comparison readable, and "
                    "horizontal is the platform default — it is also the orientation "
                    "that survives long option labels."
                    + (" A checkbox question is excluded from pie and stacked charts: "
                       "the answers do not sum to a whole."
                       if fmt == "Multi-Select" else ""),
                    stage=5,
                    inputs={"metric_type": "Categorical", "option_count": n_opts,
                            "response_format": fmt or "(unset)"},
                ),
            )

        # Rating scale default
        if metric_type in ("Standard Metric", "Custom Metric"):
            return TagResult(
                value="Horizontal Bar", source="deterministic", confidence=0.75,
                evidence=ev.rule(
                    "question.visualization_type.scale_default",
                    f"A {metric_type.lower()} that is neither the survey's headline "
                    "number nor a matrix row. Bars over the scale points show the "
                    "distribution, not just the average.",
                    stage=5,
                    inputs={"metric_type": metric_type,
                            "role_intent": role or "(unset)",
                            "response_format": fmt or "(unset)"},
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
                        "response_format": fmt or "(unset)",
                        "question_type": q.question_type},
            ),
        )


def create_tagger() -> VisualizationTypeTagger:
    return VisualizationTypeTagger()
