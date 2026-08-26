"""The platform metric vocabulary, and the one derivation both metric
dimensions read (V8, Phase 2).

Two dimensions state how a question's answers are aggregated, and they must not
be able to disagree:

* `platform_metric` — the ordered list of RMX metric codes valid for the
  question, primary first. It maps one-to-one onto the widget payload's
  `metricDetails[]` array, where position becomes `orderId`.
* `calculation_type` — the vendor-neutral analytical statement, read by
  Reporting and S5, which have no business knowing RMX codes.

The dependency is strictly one-way: `compute_platform_metrics` derives from raw
platform signals, and `coarsen_to_calculation_type` is a *coarsening* of its
result. One source of truth, so the two cannot drift the way `control_role`
drifted from `is_filterable` before V7.3 removed it.

Provenance: this is a port of RMX's `compute_metric_types` /
`get_default_metrics_for_question` as *described* by
`docs/taxonomy-v8-migration.md` Phase 2, not a line-by-line transcription of the
Angular source. The branch order, the exclusion set and the graph-vs-table split
follow that description; the numeric codes below are the ones the plan names,
with the gaps filled by position (see `METRIC_CODES`). Verify against the RMX
source before any automation posts these codes to `/Widget/insertupdate`.
"""

from __future__ import annotations

import logging

from models.survey import QuestionContext

# The platform's twenty metrics, verbatim. These strings ARE the tag values of
# `platform_metric` — keep them in step with taxonomy.yaml's allowed_values.
NPS = "NPS"
CSAT = "CSAT"
CES = "CES"
CUSTOM_METRIC = "Custom Metric"
NET_INTENT = "Net Intent"
WEIGHTED_SCORE = "Weighted Score"
PERCENT_FAVORABLE = "Percent Favorable"
COUNT = "Count"
PERCENTAGE = "Percentage"
WEIGHTED_SCORE_PCT = "Weighted Score as Percentage"
GROUP_PERCENTAGE = "Group Percentage"
GROUP_COUNT = "Group Count"
MEAN = "Mean"
MODE = "Mode"
MEDIAN = "Median"
MIN = "Min"
MAX = "Max"
SUM = "Sum"
STD_DEV = "Std Dev"
OVERALL_RANK = "Overall Rank"

# Metric name -> the two-character `metricType` the widget payload carries.
#
# The plan names twelve of these directly (NPS 01, CSAT 02, CES 03, Net Intent
# 05, Weighted Score 06, Percent Favorable 07, Count 08, Percentage 09, Group
# Percentage 11, Mean 15, Sum 20, Overall Rank 22). The remaining eight are
# filled in by position in that sequence and are marked below — a reader
# building a payload should confirm those eight against the RMX source rather
# than trusting this table.
METRIC_CODES: dict[str, str] = {
    NPS: "01",
    CSAT: "02",
    CES: "03",
    CUSTOM_METRIC: "04",          # inferred by position
    NET_INTENT: "05",
    WEIGHTED_SCORE: "06",
    PERCENT_FAVORABLE: "07",
    COUNT: "08",
    PERCENTAGE: "09",
    WEIGHTED_SCORE_PCT: "10",     # inferred by position
    GROUP_PERCENTAGE: "11",
    GROUP_COUNT: "12",            # inferred by position
    MEAN: "15",
    MODE: "16",                   # inferred by position
    MEDIAN: "17",                 # inferred by position
    MIN: "18",                    # inferred by position
    MAX: "19",                    # inferred by position
    SUM: "20",
    STD_DEV: "21",                # inferred by position
    OVERALL_RANK: "22",
}

ALL_METRICS: tuple[str, ...] = tuple(METRIC_CODES)

# The seven numeric metrics a statistical text question gets, in RMX's own
# order — Sum is the primary, which is what finally gives `calculation_type:
# Sum` a rule rather than only a platform hint.
_NUMERIC_METRICS = (SUM, MEAN, MODE, MEDIAN, MIN, MAX, STD_DEV)

# The four advanced metrics, offered for any weighted answer shape outside the
# exclusion set below. Weighted Score leads because it is the platform default
# for every grid and rating question.
_ADVANCED_METRICS = (WEIGHTED_SCORE, WEIGHTED_SCORE_PCT, PERCENT_FAVORABLE, NET_INTENT)

# Answer shapes the advanced metrics are excluded for. A checkbox answer is not
# a point on a scale — a respondent who ticks three boxes has no score — so the
# weighted family does not apply however the options are weighted.
_ADVANCED_EXCLUDED_FORMATS = frozenset({
    "Multi-Select", "Ranking", "Date", "Numeric-Open", "Open-Text",
    "Contact", "File-Upload", "Not Applicable",
})

# The band split a platform-scored metric carries (promoter / passive /
# detractor and its CSAT / CES equivalents).
_BAND_METRICS = (GROUP_PERCENTAGE, GROUP_COUNT)

# Every question that has anything to count at all gets these, last.
_UNIVERSAL_METRICS = (PERCENTAGE, COUNT)

# Formats with no analyzable answer: nothing to aggregate, so the list is empty
# and `calculation_type` falls through to Not Applicable.
_NO_METRIC_FORMATS = frozenset({
    "Open-Text", "Contact", "File-Upload", "Not Applicable",
})

# Platform `calculationType` hint (free text the survey author configured) ->
# the metric it names. Ordered longest-first inside each branch so "weighted
# score as percentage" cannot be swallowed by "weighted".
#
# The V8 fix lives here: the previous normalization mapped anything containing
# "weighted" onto Mean, discarding an explicit author choice and contradicting
# RMX, whose own default for that question is Weighted Score.
logger = logging.getLogger(__name__)


_HINT_MATCHES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("weighted score as percentage", "weighted percentage"), WEIGHTED_SCORE_PCT),
    (("weighted",), WEIGHTED_SCORE),
    (("nps", "net promoter"), NPS),
    (("net intent",), NET_INTENT),
    (("percent favorable", "percent favourable", "favorable", "favourable"),
     PERCENT_FAVORABLE),
    # "Top Box %" is the author asking for the favorable band, and the platform
    # has a metric for exactly that. Without this row the trailing "%" drops it
    # onto generic Percentage further down — better than the pre-V8 behaviour (it
    # fell all the way through to Mean) but still not what was configured.
    (("top box", "topbox"), PERCENT_FAVORABLE),
    (("group percentage",), GROUP_PERCENTAGE),
    (("group count",), GROUP_COUNT),
    (("overall rank", "rank"), OVERALL_RANK),
    (("std dev", "standard deviation"), STD_DEV),
    (("median",), MEDIAN),
    (("mode",), MODE),
    (("minimum", "min"), MIN),
    (("maximum", "max"), MAX),
    (("sum", "total"), SUM),
    (("mean", "average"), MEAN),
    (("percentage", "percent", "%"), PERCENTAGE),
    (("count",), COUNT),
)


def metric_from_platform_hint(raw: str | None) -> str | None:
    """The metric a survey author's own `calculationType` setting names, or None.

    An explicit author choice outranks every inference, so the caller puts the
    result first in the ordered list. Both dimensions read it through here, so a
    hint can never move one of them without moving the other.
    """
    if not raw:
        return None
    text = " ".join(str(raw).split()).strip().lower()
    if not text:
        return None
    for needles, metric in _HINT_MATCHES:
        if any(n in text for n in needles):
            return metric
    # The author set something this table does not know. Falling through to the
    # shape rules is right — guessing at their meaning would be worse — but doing
    # it silently is how "Top Box %" spent eleven surveys reported as Mean.
    logger.warning("calculation_type_hint_unrecognized",
                   extra={"platform_calculation_type": raw})
    return None


def _has_weights(q: QuestionContext) -> bool:
    return any(o.weight is not None for o in q.answer_options)


def _dedupe(metrics) -> list[str]:
    """Preserve first-seen order; position in this list becomes `orderId`."""
    seen: set[str] = set()
    out: list[str] = []
    for m in metrics:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def compute_platform_metrics(
    q: QuestionContext,
    response_format: str,
    *,
    honor_platform_hint: bool = True,
) -> list[str]:
    """The platform metrics valid for this question, primary first.

    One list rather than the platform's graph/table pair: the graph primary
    leads, and the metrics only a table can show (band splits, Overall Rank)
    follow it. A composer building a table reads the whole list; one building a
    chart takes the head. Keeping them in one ordered field is what lets
    `metricDetails[].orderId` be filled straight from the index.
    """
    if response_format in _NO_METRIC_FORMATS:
        return []

    hint = metric_from_platform_hint(q.calculation_type) if honor_platform_hint else None

    if response_format == "Date":
        # RMX allows nothing else on a date question, and an author hint cannot
        # widen that — there is no other aggregate a date column supports.
        return [COUNT]

    if response_format == "Numeric-Open":
        return _dedupe([hint, *_NUMERIC_METRICS])

    if response_format == "Ranking":
        # Weighted Score on graphs; Percentage / Count / Weighted Score /
        # Overall Rank on tables. There is no mean-rank metric in RMX.
        return _dedupe([hint, WEIGHTED_SCORE, *_UNIVERSAL_METRICS, OVERALL_RANK])

    metrics: list[str] = [hint] if hint else []

    # The rs_type-specific metric leads whenever the platform scores the
    # question, and its band split follows — a table on an NPS question emits
    # [NPS, Group Percentage] and gets both rows, exactly as the manual UI does.
    if q.rs_type == 2:
        metrics += [NPS, *_BAND_METRICS]
    elif q.rs_type == 3:
        metrics += [CES, *_BAND_METRICS]
    elif q.rs_type == 4:
        metrics += [CSAT, *_BAND_METRICS]
    elif q.is_custom_metric:
        metrics += [CUSTOM_METRIC, *_BAND_METRICS]

    # The advanced family, for any weighted answer shape the platform does not
    # exclude. This is where Weighted Score — the RMX default for every grid and
    # rating question, and the single most-used code — enters.
    if response_format not in _ADVANCED_EXCLUDED_FORMATS and _has_weights(q):
        metrics += list(_ADVANCED_METRICS)

    metrics += list(_UNIVERSAL_METRICS)
    return _dedupe(metrics)


# platform_metric -> the vendor-neutral `calculation_type` it coarsens to.
#
# Mode / Median / Min / Max / Std Dev have no analytical value of their own in
# our vocabulary and never lead a list, so they fold into Mean. Recording that
# here rather than leaving them unmapped keeps the coarsening total.
_COARSEN: dict[str, str] = {
    NPS: "NPS Score",
    CSAT: "Weighted Score",
    CES: "Weighted Score",
    CUSTOM_METRIC: "Weighted Score",
    NET_INTENT: "Net Intent",
    WEIGHTED_SCORE: "Weighted Score",
    WEIGHTED_SCORE_PCT: "Weighted Score",
    PERCENT_FAVORABLE: "Percent Favorable",
    COUNT: "Count",
    PERCENTAGE: "Percentage",
    GROUP_PERCENTAGE: "Distribution",
    GROUP_COUNT: "Distribution",
    MEAN: "Mean",
    MODE: "Mean",
    MEDIAN: "Mean",
    MIN: "Mean",
    MAX: "Mean",
    STD_DEV: "Mean",
    SUM: "Sum",
    OVERALL_RANK: "Mean Rank",
}

_SELECT_FORMATS = frozenset({"Single-Select", "Multi-Select", "Hidden-Select"})
_SCALE_FORMATS = frozenset({"Rating-Scale", "Effort-Scale", "Matrix-Row"})


def coarsen_to_calculation_type(
    metrics: list[str], response_format: str, option_count: int
) -> str:
    """Fold the ordered metric list down to one analytical statement.

    Reads the same raw inputs as `compute_platform_metrics` rather than only its
    output, because three answer shapes need a different word from the one their
    primary platform metric supplies. That is translation, not disagreement —
    both dimensions still come from this single function, so neither can drift
    from the other:

    * **Ranking.** Its primary metric is Weighted Score, but what a report says
      about a ranking question is the average position — `Mean Rank`. The
      remapped platform target lives in `platform_metric`; the analytical name
      stays what it always was.
    * **A choice list with three or more options.** Its primary metric is
      Percentage, but no single percentage represents it — the shape across all
      options is the result, which is `Distribution`. Two options are the one
      case where a single percentage does tell the whole story.
    * **An UNWEIGHTED rating scale or grid.** The platform can only count it,
      because a weighted score needs weights — so `platform_metric` correctly
      says Percentage and Count. But the scale points are still ordered, and
      what Reporting does with an ordinal scale is average them. `Mean` is the
      honest analytical word, and this is the branch that keeps it reachable now
      that weighted questions and CSAT/CES have stopped being folded into it.

    Everything else is a straight lookup.
    """
    if not metrics:
        return "Not Applicable"
    if response_format == "Ranking":
        return "Mean Rank"
    if response_format in _SELECT_FORMATS:
        return "Percentage" if option_count == 2 else "Distribution"
    if response_format in _SCALE_FORMATS and metrics[0] == PERCENTAGE:
        return "Mean"
    return _COARSEN.get(metrics[0], "Not Applicable")
