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
                             evidence="Content message")
        value = derive_capability(question)[self.tag_dimension]
        return TagResult(value=value, source="deterministic", confidence=1.0,
                         evidence=f"Derived from question_type={question.question_type}, "
                                  f"rs_type={question.rs_type}")


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
