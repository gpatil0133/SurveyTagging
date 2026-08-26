"""Dashboard capability taggers (V7, reshaped in V8) — the per-question
capability layer that the downstream (LLM-driven) dashboard-composition service
consumes.

Five Stage-3 deterministic dimensions, all derived from raw QuestionContext
signals (rs_type, question_type, question_sub_type, is_multi, answer-option
weights). They state what a question is CAPABLE of; the dashboard service
confirms data VIABILITY (response volume, null rate, distinct-value count) and
does the contextual selection / pairing / layout.

    response_format ....... canonical answer shape (the anchor)
    scale_of_measurement .. Nominal / Ordinal / Interval / Ratio
    cardinality_class ..... Binary / Low / High / Continuous / Free-Text
    widget_compatibility .. SET of valid widgets (multi-label)
    crosstab_axis_role .... Row / Column / Both / None (table widgets)

The derivation lives once in `derive_capability()`; the five thin tagger
classes each return their slice. `create_tagger()` returns all five so the
registry auto-registers them from this single module.

V8 split the anchor. `_response_format` branched on `question_type` alone and so
collapsed five distinct question shapes into `Open-Text`: a numeric "how many
days since your last visit?" was tagged Open-Text -> Unstructured -> Free-Text
-> Word Cloud, where the platform would offer it Sum, Mean, Mode, Median, Min,
Max and Std Dev. `question_sub_type` was already on the model and already read
by `role_intent` and `data_sensitivity`; it is now read here too, through
`taggers/_sub_types.py` so the three cannot drift. Three formats came out of the
split — `Date`, `Numeric-Open`, `File-Upload` — and `Contact` widened to absorb
the email sub-type, whose capability (an identifier, table only) is identical to
a CS block's.

V8 also rewrote the widget vocabulary. Orientation is not cosmetic on this
platform — horizontal and vertical bars are separate codes, and for a grid
question the two stacked variants are the *only* legal charts — so `Bar Chart`
and `Stacked Bar Chart` split, `Score Card` took the platform's own name
(`Number`), and six real codes that we could never recommend were added. Four
values went the other way: `Trend Line` is a display mode rather than a chart,
`Ranking Bar` is a horizontal bar with a Weighted Score metric, and `Heat Map`,
`Distribution Chart`, `Sentiment` and `Map` have no chart code behind them at
all.

V7.3 removed a sixth dimension, `control_role`. It restated `is_filterable`,
`is_segmentable` and "is this Open-Text" in one multi-label field, which meant
three dimensions had to be kept in agreement and — because this module is Stage 3
and cannot see `role_intent` — it silently disagreed with `is_segmentable` on any
question where role_intent was the deciding signal. What a dashboard composer
needs from it is now read directly:

    filter dropdown  ->  is_filterable == "Yes"
    segment picker   ->  is_segmentable == "Yes"
    text search      ->  response_format == "Open-Text"
    date filter      ->  response_format == "Date"   (V8: now derivable)

See the `derived_controls` block on `crosstab_axis_role` in taxonomy.yaml, which
states that mapping for consumers.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers import _sub_types
from taggers._metric_utils import is_platform_metric
from taggers._platform_metrics import compute_platform_metrics
from taggers.base import QuestionTagger

# Raw question_type code groups (consistent with metric_type / is_filterable).
_SELECT_TYPES = {"L", "R", "C", "SR", "ML"}
_GRID_TYPES = {"GR", "GC", "RG", "GQ"}
_RATING_TYPES = {"RS", "RT", "RW"}   # RK (ranking) handled separately
_METRIC_FORMATS = {"NPS-Scale", "Rating-Scale", "Effort-Scale", "Ranking"}

# Formats whose answer rolls up to a score or a measurement rather than into
# option buckets. Wider than `_METRIC_FORMATS`: the two new numeric shapes are
# continuous without being scales you can facet on.
_CONTINUOUS_FORMATS = _METRIC_FORMATS | {"Matrix-Row", "Date", "Numeric-Open"}

# --- Chart vocabulary -------------------------------------------------------
#
# Every name here is a real platform chart code; nothing below is a concept we
# invented. `Word Cloud` and `Scatter Plot` are deliberately unreachable while
# remaining allowed values on the dimension — the first needs open-ended widget
# type 20, which has no creator, and the second only ever arrived through LLM
# refinement of `visualization_type`.
#
# The set is computed rather than tabulated, because the platform's allowed
# charts genuinely flip on the METRIC: the count/percentage family unlocks
# everything that divides a whole (pie, donut, stacked, bubble, radar), while a
# scored metric collapses the set to bars plus gauge and number. Stating a
# per-format list as if it were metric-independent could not be honest — an
# unweighted rating scale would claim a gauge it has no score to fill.
#
# So `widget_compatibility` is the UNION of the chart families over the
# question's own `platform_metric` list, intersected with the platform's
# structural restriction for its answer shape. Two rules, one derivation, and
# the metric list it reads is the same one `platform_metric` publishes. The
# `metric_constraints` block on the dimension in taxonomy.yaml states the
# families for a consumer narrowing the union back down for one widget.

# Canonical chart order — most-default first. Applied to the computed set so
# identical input always yields an identical list.
_CHART_ORDER: tuple[str, ...] = (
    "Number", "Gauge",
    "Horizontal Bar", "Vertical Bar",
    "Horizontal Stacked Bar", "Vertical Stacked Bar",
    "Clustered Horizontal Bar", "Clustered Vertical Bar",
    "Line Chart", "Area",
    "Pie Chart", "Donut", "Radar", "Bubble",
    "Scatter Plot", "Word Cloud",
    "Table",
)

# A scored metric produces ONE number per series: show it big, put a needle on
# it, or plot it over the scale points and over time.
_SCORED_CHARTS = frozenset({
    "Number", "Gauge", "Horizontal Bar", "Vertical Bar", "Line Chart", "Area",
    "Table",
})

# A count/percentage metric produces a distribution across options, which is
# what every chart that divides a whole needs.
_COUNT_CHARTS = frozenset({
    "Horizontal Bar", "Vertical Bar",
    "Horizontal Stacked Bar", "Vertical Stacked Bar",
    "Clustered Horizontal Bar", "Clustered Vertical Bar",
    "Line Chart", "Area", "Pie Chart", "Donut", "Radar", "Bubble", "Table",
})

# Which family a platform metric belongs to. Everything not listed here as a
# count metric is scored — the numeric seven included, since each of them is
# also one number per series.
_COUNT_FAMILY_METRICS = frozenset({
    "Count", "Percentage", "Group Percentage", "Group Count",
})

# Charts that need a comparison AXIS — several attributes, or a segment — and
# not just several answer options. A single scale question is one series, so a
# radar with one spoke or a clustered bar with one cluster is not a chart of
# anything. Excluded from the scale formats below for that reason.
_MULTI_SERIES_CHARTS = frozenset({
    "Radar", "Bubble", "Clustered Horizontal Bar", "Clustered Vertical Bar",
})

# What a single scored scale question permits: the count family minus the
# charts that need more than one series, plus the scored family.
_SCALE_CHARTS = frozenset(
    (_SCORED_CHARTS | _COUNT_CHARTS) - _MULTI_SERIES_CHARTS
)

# What the platform structurally permits for an answer shape, whatever its
# metrics. `None` means "no restriction beyond the metric families".
_FORMAT_RESTRICTION: dict[str, frozenset[str] | None] = {
    "NPS-Scale": _SCALE_CHARTS,
    "Rating-Scale": _SCALE_CHARTS,
    "Effort-Scale": _SCALE_CHARTS,
    "Single-Select": None,
    # Checkbox EXCLUDES stacked, pie and donut: the answers do not sum to a
    # whole, so any chart that divides one misrepresents the data.
    "Multi-Select": frozenset(_COUNT_CHARTS - {
        "Pie Chart", "Donut", "Horizontal Stacked Bar", "Vertical Stacked Bar",
    }),
    # A routing question is segmentable but never a headline widget.
    "Hidden-Select": frozenset({"Horizontal Bar", "Vertical Bar", "Table"}),
    # The two stacked variants are the ONLY legal charts for a grid question,
    # which is precisely why guessing the orientation was most costly here.
    "Matrix-Row": frozenset({
        "Horizontal Stacked Bar", "Vertical Stacked Bar", "Table",
    }),
    # Bubble is available on ranking questions; the old `Ranking Bar` was only
    # ever a horizontal bar carrying a Weighted Score metric.
    "Ranking": frozenset({"Horizontal Bar", "Vertical Bar", "Bubble", "Table"}),
    # Count is the only metric a date question supports, so what is chartable is
    # a count per period.
    "Date": frozenset({
        "Horizontal Bar", "Vertical Bar", "Line Chart", "Area", "Table",
    }),
    "Numeric-Open": frozenset({
        "Number", "Horizontal Bar", "Vertical Bar", "Line Chart", "Area", "Table",
    }),
    # Table only. A word cloud needs open-ended widget type 20 and sentiment
    # needs tag-sourced widgets (sourceType "02"); neither has a creator.
    "Open-Text": frozenset({"Table"}),
    "Contact": frozenset({"Table"}),
    # There is no analyzable answer at all — before V8 a file upload became a
    # word cloud of filenames.
    "File-Upload": frozenset(),
    "Not Applicable": frozenset(),
}


def _widgets(fmt: str, q: QuestionContext) -> list[str]:
    """Every widget that can validly render this question.

    The union of the chart families its platform metrics unlock, narrowed to
    what the platform structurally permits for its answer shape.
    """
    restriction = _FORMAT_RESTRICTION.get(fmt, frozenset())
    if restriction is not None and not restriction:
        return []

    metrics = compute_platform_metrics(q, fmt)
    if not metrics:
        # No metric means nothing to plot. A table still renders the answers
        # themselves, which is why Open-Text and Contact are not empty.
        allowed = {"Table"} if restriction is None or "Table" in restriction else set()
    else:
        allowed = set()
        for metric in metrics:
            allowed |= (_COUNT_CHARTS if metric in _COUNT_FAMILY_METRICS
                        else _SCORED_CHARTS)
    if restriction is not None:
        allowed &= restriction
    return [c for c in _CHART_ORDER if c in allowed]


def _response_format(q: QuestionContext) -> str:
    """Canonical answer shape. rs_type wins over question_type, and for a text
    question the sub-type wins over the bare `T`.

    The sub-type branch is the V8 correction: `T` alone says only "not a choice
    list", and treating it as prose swept date, email, numeric and file-upload
    questions into `Open-Text` — where every downstream dimension inherited the
    error. The codes are named once in `taggers/_sub_types.py`.
    """
    if q.rs_type == 2:
        return "NPS-Scale"
    if q.rs_type == 3:
        return "Effort-Scale"
    if q.rs_type == 4:
        return "Rating-Scale"

    qt = q.question_type
    if qt == "T":
        st = q.question_sub_type
        if st in _sub_types.EMAIL:
            # Same capability as a CS block: an identifier, table only.
            return "Contact"
        if st in _sub_types.DATE:
            return "Date"
        if st in _sub_types.FILE_UPLOAD:
            return "File-Upload"
        if st in _sub_types.NUMERIC:
            return "Numeric-Open"
        return "Open-Text"
    if qt == "CS":
        return "Contact"
    if qt == "RK":
        return "Ranking"
    if qt in _RATING_TYPES:
        return "Rating-Scale"
    if qt in _GRID_TYPES:
        return "Matrix-Row"
    if qt == "HR":
        return "Hidden-Select"
    if qt in _SELECT_TYPES:
        return "Multi-Select" if q.is_multi else "Single-Select"
    return "Not Applicable"


def _has_weights(q: QuestionContext) -> bool:
    return any(o.weight is not None for o in q.answer_options)


def _scale(fmt: str, q: QuestionContext) -> str:
    if fmt in ("NPS-Scale", "Date"):
        # Date joins NPS on the Interval branch: its points are ordered with
        # meaningful intervals (a week is a week wherever it falls) but there is
        # no true zero, so differences are valid and ratios are not.
        return "Interval"
    if fmt in ("Rating-Scale", "Effort-Scale", "Ranking"):
        return "Ordinal"
    if fmt in ("Single-Select", "Multi-Select", "Hidden-Select"):
        return "Nominal"
    if fmt == "Matrix-Row":
        return "Ordinal" if _has_weights(q) else "Nominal"
    if fmt == "Numeric-Open":
        # The first and only source of a true Ratio scale. `Ratio` was emitted
        # by nothing in `taggers/` before V8 — a dead value the Phase 1 split
        # brought to life rather than a reason to delete it.
        return "Ratio"
    if fmt == "Open-Text":
        return "Unstructured"
    # Contact and File-Upload: an identifier or an attachment measures nothing.
    return "N/A"


def _cardinality(fmt: str, q: QuestionContext) -> str:
    if fmt == "Open-Text":
        return "Free-Text"
    if fmt in ("Contact", "File-Upload", "Not Applicable"):
        return "N/A"
    if fmt in _CONTINUOUS_FORMATS:
        return "Continuous"
    n = q.option_count
    if n == 2:
        return "Binary"
    if 3 <= n <= 7:
        return "Low"
    if n >= 8:
        return "High"
    return "N/A"


def _crosstab_axis(fmt: str, q: QuestionContext) -> str:
    """Which axis of a cross-tab this question can occupy.

    This used to be derived from `control_role`, a multi-label dimension that has
    since been removed. Substituting its definition in collapses the whole thing
    to two questions, because `control_role` gave EVERY select format
    Dropdown-Filter unconditionally — which alone made a non-metric a grouping.
    Its `Segment-Control` refinement (unweighted options, a 2-15 bucket window,
    the Hidden-Select special case) could therefore never change the answer for a
    select format, and mattered only on the metric branch, where the test reduces
    to `is_platform_metric`. Checked exhaustively against the old pair over every
    format x rs_type x option-count x weighting combination: identical.

    A metric you can merely filter on stays Column-Eligible — it is still what
    gets compared. Only the platform's bands put a metric on the row axis;
    without that rule every metric would read "Both" and "put this in the cells"
    would become inexpressible, which is the one instruction a dashboard composer
    needs from this dimension.

    V8 note — the three new formats all read None, and `Numeric-Open` is the one
    worth stating out loud. It is genuinely a measure, and Column-Eligible is
    tempting. But Phase 1 decided explicitly that an unbounded numeric answer is
    not a bounded scale: it builds no facet, so `bounded_scale_kind` returns None
    for it and `is_filterable` says No. The invariant recorded in
    `taggers/_metric_utils.py` is that this dimension and `is_filterable` must
    not state opposite things about whether an answer lands on a scale — so
    Numeric-Open reads None on both sides rather than measuring on one and not
    the other. Flip both together, or neither. (Cross-tab widget type 40 has no
    creator today either way.)
    """
    if fmt in _METRIC_FORMATS or fmt == "Matrix-Row":
        return "Both" if is_platform_metric(q) else "Column-Eligible"
    if fmt in ("Single-Select", "Multi-Select", "Hidden-Select"):
        return "Row-Eligible"
    return "None"


def derive_capability(q: QuestionContext) -> dict[str, object]:
    """Compute all five capability values for one question in a single pass."""
    fmt = _response_format(q)
    return {
        "response_format": fmt,
        "scale_of_measurement": _scale(fmt, q),
        "cardinality_class": _cardinality(fmt, q),
        "widget_compatibility": _widgets(fmt, q),
        "crosstab_axis_role": _crosstab_axis(fmt, q),
    }


def explain_capability(q: QuestionContext, dimension: str, value: object) -> str:
    """One sentence saying why `dimension` came out as `value` for this question.

    All six dimensions cascade off `response_format`, so each explanation names
    that anchor rather than repeating the raw type codes — a reader following
    the chain can see where a wrong answer entered it.
    """
    fmt = _response_format(q)
    anchor = (f"rs_type={q.rs_type}" if q.rs_type in (2, 3, 4)
              else f"question_type={q.question_type!r}")
    if q.question_type == "T" and q.rs_type not in (2, 3, 4):
        anchor += f", question_sub_type={q.question_sub_type}"

    if dimension == "response_format":
        # The four sub-type-decided shapes explain themselves, because "it is a
        # T question" is exactly the reasoning V8 replaced.
        sub_type_note = {
            "Date": ("Sub-type 1 is the platform's date picker, not free prose. The "
                     "platform restricts date questions to the Count metric, and this "
                     "is also the value a dashboard's date filter keys off."),
            "Numeric-Open": ("The sub-type marks this as a statistical text field: the "
                             "answer is a number, so it supports Sum, Mean, Mode, "
                             "Median, Min, Max and Std Dev, and it is the only shape "
                             "that lands on a true Ratio scale."),
            "File-Upload": ("Sub-type 71 is a file upload. There is no analyzable "
                            "answer at all — before this split it became a word cloud "
                            "of filenames."),
            "Contact": ("An identifier rather than an opinion — a contact block, or the "
                        "email-validated text field, which has the same capability: a "
                        "table, and nothing else."),
        }.get(str(value))
        base = (f"The canonical answer shape is {value}, read off {anchor}. A platform "
                "metric flag (rs_type) always wins over the question type, and for a "
                "text question the sub-type wins over the bare `T`, because the same "
                "widget type is used for scored, unscored, dated and numeric "
                "questions alike. Every other capability dimension cascades off "
                "this one.")
        return f"{base} {sub_type_note}" if sub_type_note else base
    if dimension == "scale_of_measurement":
        if value == "Unstructured":
            return (f"A {fmt} answer is prose — it has no measurement scale, so nothing "
                    "can be ordered or averaged.")
        if value == "N/A":
            return (f"A {fmt} answer carries no measurement at all (identity capture, "
                    "an attachment, or an unrecognized question type).")
        return (f"A {fmt} answer is {value}: "
                + {"Interval": "its points are ordered with meaningful intervals but "
                               "there is no true zero, so differences are valid and "
                               "ratios are not",
                   "Ratio": "it is a real number with a true zero, so every statistic "
                            "applies — sums and ratios included",
                   "Ordinal": "its points are ordered but not evenly spaced, so "
                              "medians and rank tests are valid but means are a "
                              "convenient approximation",
                   "Nominal": "its options are labels with no order, so only counts "
                              "and proportions are valid"}.get(str(value), str(value))
                + ". This is what decides which statistics a widget may show.")
    if dimension == "cardinality_class":
        if value == "Continuous":
            return (f"A {fmt} answer rolls up to a score or a measurement rather than "
                    "falling into buckets, so it is treated as continuous for widget "
                    "selection.")
        if value == "Free-Text":
            return f"A {fmt} answer is unbounded prose — there is no option count."
        if value == "N/A":
            return f"A {fmt} answer has no option set to count."
        return (f"The question has {q.option_count} answer option(s), which puts it in "
                f"the {value} band (2 = Binary, 3-7 = Low, 8+ = High). Cardinality is "
                "what decides whether a chart stays readable.")
    if dimension == "widget_compatibility":
        widgets = value if isinstance(value, list) else []
        if not widgets:
            return (f"A {fmt} answer supports no dashboard widget — there is nothing to "
                    "plot.")
        note = {
            "Matrix-Row": (" The two stacked variants are the only charts the platform "
                           "permits on a grid question — there is no third option to "
                           "choose badly between."),
            "Multi-Select": (" A checkbox answer excludes stacked bars, pie and donut: "
                             "the responses do not sum to a whole, so any chart that "
                             "divides one misrepresents the data."),
            "Open-Text": (" Table only. A word cloud needs open-ended widget type 20 and "
                          "sentiment needs tag-sourced widgets; neither has a creator "
                          "on the platform yet."),
        }.get(fmt, "")
        return (f"A {fmt} answer can be rendered as any of {len(widgets)} widget(s): "
                f"{', '.join(str(w) for w in widgets)}. This is the union over the "
                "question's valid metrics — a count/percentage metric unlocks more of "
                "them than a scored one does — and it is structural validity only: the "
                "dashboard service still checks whether the response volume and "
                "distinct-value count make each one worth showing." + note)
    if dimension == "crosstab_axis_role":
        return {
            "Both": (f"A {fmt} answer measures something, and the platform bands it "
                     "(rs_type 2/3/4 or is_custom_metric) so those bands can group the "
                     "rest of the survey too — it sits on either axis of a cross-tab."),
            "Column-Eligible": (f"A {fmt} answer is a measure, so it belongs in the "
                                "cells/columns of a cross-tab — it is what gets "
                                "compared, not what does the comparing. An unbanded "
                                "scale stays here: you can filter on it without having "
                                "an agreed grouping to break other questions out by."),
            "Row-Eligible": (f"A {fmt} answer is a bounded set of choices rather than a "
                             "measure, so it belongs on the row axis — it is what other "
                             "questions get broken out by."),
            "None": (
                f"A {fmt} answer measures a real quantity, but on an unbounded scale — "
                "it builds no facet and the platform bands it into nothing, so there is "
                "no axis for it to occupy. is_filterable says No for the same reason; "
                "the two are decided together."
                if fmt == "Numeric-Open" else
                f"A {fmt} answer neither measures nor groups, so it has no place "
                "in a cross-tab."
            ),
        }.get(str(value), f"Derived from response_format {fmt}.")
    return f"Derived from response_format {fmt} ({anchor})."


class _CapabilityTagger(QuestionTagger):
    """Base for the six capability taggers. Each subclass sets `tag_dimension`
    and whether its value is multi-label (list) or scalar."""

    stage = 3
    source_type = "deterministic"
    _multi = False

    @property
    def skip_value(self):
        # Mirrors `_multi`: the list-valued dimensions skip to [], the scalar ones
        # to None. Same split the subclasses already declare.
        return [] if self._multi else None

    @property
    def name(self) -> str:
        return f"question.{self.tag_dimension}"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        value = derive_capability(question)[self.tag_dimension]
        return TagResult(
            value=value, source="deterministic", confidence=1.0,
            evidence=ev.rule(
                f"question.{self.tag_dimension}.derived_capability",
                explain_capability(question, self.tag_dimension, value),
                stage=3,
                inputs={"question_type": question.question_type,
                        "question_sub_type": question.question_sub_type,
                        "rs_type": question.rs_type,
                        "is_multi": question.is_multi,
                        "option_count": question.option_count,
                        "response_format": _response_format(question)},
            ),
        )


class ResponseFormatTagger(_CapabilityTagger):
    tag_dimension = "response_format"


class ScaleOfMeasurementTagger(_CapabilityTagger):
    tag_dimension = "scale_of_measurement"


class CardinalityClassTagger(_CapabilityTagger):
    tag_dimension = "cardinality_class"


class WidgetCompatibilityTagger(_CapabilityTagger):
    tag_dimension = "widget_compatibility"
    _multi = True


class CrosstabAxisRoleTagger(_CapabilityTagger):
    tag_dimension = "crosstab_axis_role"


def create_tagger() -> list[QuestionTagger]:
    return [
        ResponseFormatTagger(),
        ScaleOfMeasurementTagger(),
        CardinalityClassTagger(),
        WidgetCompatibilityTagger(),
        CrosstabAxisRoleTagger(),
    ]
