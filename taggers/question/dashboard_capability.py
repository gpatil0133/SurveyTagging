"""Dashboard capability taggers (V7) — the per-question capability layer that
the downstream (LLM-driven) dashboard-composition service consumes.

Six Stage-3 deterministic dimensions, all derived from raw QuestionContext
signals (rs_type, question_type, is_multi, answer-option weights). They state
what a question is CAPABLE of; the dashboard service confirms data VIABILITY
(response volume, null rate, distinct-value count) and does the contextual
selection / pairing / layout.

    response_format ....... canonical answer shape (the anchor)
    scale_of_measurement .. Nominal / Ordinal / Interval / Ratio
    cardinality_class ..... Binary / Low / High / Continuous / Free-Text
    widget_compatibility .. SET of valid widgets (multi-label)
    control_role .......... filter / segment control roles (multi-label)
    crosstab_axis_role .... Row / Column / Both / None (table widgets)

The derivation lives once in `derive_capability()`; the six thin tagger
classes each return their slice. `create_tagger()` returns all six so the
registry auto-registers them from this single module.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# Raw question_type code groups (consistent with metric_type / is_filterable).
_SELECT_TYPES = {"L", "R", "C", "SR", "ML"}
_GRID_TYPES = {"GR", "GC", "RG", "GQ"}
_RATING_TYPES = {"RS", "RT", "RW"}   # RK (ranking) handled separately
_METRIC_FORMATS = {"NPS-Scale", "Rating-Scale", "Effort-Scale", "Ranking"}

# response_format -> valid widget set (values must match taxonomy allowed_values).
_WIDGETS: dict[str, list[str]] = {
    "NPS-Scale": ["Score Card", "Gauge", "Trend Line", "Distribution Chart", "Bar Chart"],
    "Rating-Scale": ["Score Card", "Gauge", "Bar Chart", "Trend Line", "Distribution Chart"],
    "Effort-Scale": ["Score Card", "Gauge", "Bar Chart", "Trend Line", "Distribution Chart"],
    "Ranking": ["Ranking Bar", "Bar Chart", "Table"],
    "Single-Select": ["Bar Chart", "Pie Chart", "Table", "Distribution Chart"],
    "Multi-Select": ["Bar Chart", "Stacked Bar Chart", "Table"],
    "Hidden-Select": ["Table", "Bar Chart"],
    "Matrix-Row": ["Heat Map", "Stacked Bar Chart", "Bar Chart", "Table"],
    "Open-Text": ["Word Cloud", "Sentiment", "Table"],
    "Contact": ["Table"],
    "Not Applicable": [],
}


def _response_format(q: QuestionContext) -> str:
    """Canonical answer shape. rs_type wins over question_type."""
    if q.rs_type == 2:
        return "NPS-Scale"
    if q.rs_type == 3:
        return "Effort-Scale"
    if q.rs_type == 4:
        return "Rating-Scale"

    qt = q.question_type
    if qt == "T":
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
    if fmt == "NPS-Scale":
        return "Interval"
    if fmt in ("Rating-Scale", "Effort-Scale", "Ranking"):
        return "Ordinal"
    if fmt in ("Single-Select", "Multi-Select", "Hidden-Select"):
        return "Nominal"
    if fmt == "Matrix-Row":
        return "Ordinal" if _has_weights(q) else "Nominal"
    if fmt == "Open-Text":
        return "Unstructured"
    return "N/A"


def _cardinality(fmt: str, q: QuestionContext) -> str:
    if fmt == "Open-Text":
        return "Free-Text"
    if fmt in ("Contact", "Not Applicable"):
        return "N/A"
    if fmt in _METRIC_FORMATS or fmt == "Matrix-Row":
        return "Continuous"
    n = q.option_count
    if n == 2:
        return "Binary"
    if 3 <= n <= 7:
        return "Low"
    if n >= 8:
        return "High"
    return "N/A"


def _control_roles(fmt: str, q: QuestionContext) -> list[str]:
    """Which dashboard control roles this question can play (structural only;
    the dashboard service confirms data viability)."""
    roles: list[str] = []
    if fmt in ("Single-Select", "Multi-Select", "Hidden-Select"):
        roles.append("Dropdown-Filter")
        # Segment-eligible: unweighted, feasible bucket count (mirror is_segmentable).
        if not _has_weights(q) and 2 <= q.option_count <= 15:
            roles.append("Segment-Control")
        if fmt == "Hidden-Select":  # routing question -> always a segmenter
            if "Segment-Control" not in roles:
                roles.append("Segment-Control")
    elif fmt == "Open-Text":
        roles.append("Search-Filter")
    return roles


def _crosstab_axis(fmt: str, roles: list[str]) -> str:
    is_metric = fmt in _METRIC_FORMATS or fmt == "Matrix-Row"
    is_grouping = "Segment-Control" in roles or "Dropdown-Filter" in roles
    if is_metric and is_grouping:
        return "Both"
    if is_metric:
        return "Column-Eligible"
    if is_grouping:
        return "Row-Eligible"
    return "None"


def derive_capability(q: QuestionContext) -> dict[str, object]:
    """Compute all six capability values for one question in a single pass."""
    fmt = _response_format(q)
    roles = _control_roles(fmt, q)
    return {
        "response_format": fmt,
        "scale_of_measurement": _scale(fmt, q),
        "cardinality_class": _cardinality(fmt, q),
        "widget_compatibility": list(_WIDGETS.get(fmt, [])),
        "control_role": roles,
        "crosstab_axis_role": _crosstab_axis(fmt, roles),
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

    if dimension == "response_format":
        return (f"The canonical answer shape is {value}, read off {anchor}. A platform "
                "metric flag (rs_type) always wins over the question type, because the "
                "same widget type is used for both scored and unscored questions. "
                "Every other capability dimension cascades off this one.")
    if dimension == "scale_of_measurement":
        if value == "Unstructured":
            return (f"A {fmt} answer is prose — it has no measurement scale, so nothing "
                    "can be ordered or averaged.")
        if value == "N/A":
            return (f"A {fmt} answer carries no measurement at all (identity capture, "
                    "or an unrecognized question type).")
        return (f"A {fmt} answer is {value}: "
                + {"Interval": "its points are evenly spaced, so differences are "
                               "meaningful and a mean is valid",
                   "Ordinal": "its points are ordered but not evenly spaced, so "
                              "medians and rank tests are valid but means are a "
                              "convenient approximation",
                   "Nominal": "its options are labels with no order, so only counts "
                              "and proportions are valid"}.get(str(value), str(value))
                + ". This is what decides which statistics a widget may show.")
    if dimension == "cardinality_class":
        if value == "Continuous":
            return (f"A {fmt} answer rolls up to a score rather than falling into "
                    "buckets, so it is treated as continuous for widget selection.")
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
        return (f"A {fmt} answer can be rendered as any of {len(widgets)} widget(s): "
                f"{', '.join(str(w) for w in widgets)}. This is the structural "
                "shortlist only; the dashboard service still checks whether the "
                "response volume and distinct-value count make each one worth showing.")
    if dimension == "control_role":
        roles = value if isinstance(value, list) else []
        if not roles:
            return (f"A {fmt} answer cannot drive a dashboard control — it is something "
                    "you filter, not something you filter by.")
        return (f"A {fmt} answer can act as {', '.join(str(r) for r in roles)}. "
                "Segment-Control additionally requires unweighted options and 2-15 of "
                "them, so the buckets stay populated; a routing question always "
                "qualifies.")
    if dimension == "crosstab_axis_role":
        return {
            "Both": (f"A {fmt} answer both measures something and groups respondents, "
                     "so it can sit on either axis of a cross-tab."),
            "Column-Eligible": (f"A {fmt} answer is a measure, so it belongs in the "
                                "cells/columns of a cross-tab — it is what gets "
                                "compared, not what does the comparing."),
            "Row-Eligible": (f"A {fmt} answer groups respondents, so it belongs on the "
                             "row axis — it is what other questions get broken out by."),
            "None": (f"A {fmt} answer neither measures nor groups, so it has no place "
                     "in a cross-tab."),
        }.get(str(value), f"Derived from response_format {fmt}.")
    return f"Derived from response_format {fmt} ({anchor})."


class _CapabilityTagger(QuestionTagger):
    """Base for the six capability taggers. Each subclass sets `tag_dimension`
    and whether its value is multi-label (list) or scalar."""

    stage = 3
    source_type = "deterministic"
    _multi = False

    @property
    def name(self) -> str:
        return f"question.{self.tag_dimension}"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            empty = [] if self._multi else None
            return TagResult(value=empty, source="deterministic", status="skipped",
                             evidence=ev.content_message(self.tag_dimension, stage=3))
        value = derive_capability(question)[self.tag_dimension]
        return TagResult(
            value=value, source="deterministic", confidence=1.0,
            evidence=ev.rule(
                f"question.{self.tag_dimension}.derived_capability",
                explain_capability(question, self.tag_dimension, value),
                stage=3,
                inputs={"question_type": question.question_type,
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


class ControlRoleTagger(_CapabilityTagger):
    tag_dimension = "control_role"
    _multi = True


class CrosstabAxisRoleTagger(_CapabilityTagger):
    tag_dimension = "crosstab_axis_role"


def create_tagger() -> list[QuestionTagger]:
    return [
        ResponseFormatTagger(),
        ScaleOfMeasurementTagger(),
        CardinalityClassTagger(),
        WidgetCompatibilityTagger(),
        ControlRoleTagger(),
        CrosstabAxisRoleTagger(),
    ]
