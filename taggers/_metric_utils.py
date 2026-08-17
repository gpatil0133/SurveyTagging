"""Shared metric helpers for question-level taggers.

Two different questions get asked about metrics, and they have different
answers — keep them apart:

* `is_journey_eligible_metric` — "should this question be placed on the tenant
  journey?" Journey placement is reserved for the four kinds we track in the
  Custom Journey: NPS, CSAT, CES (identified by rs_type) or a platform-flagged
  custom metric (`is_custom_metric`). Heuristic matrix-type (GR/GC/RG/GQ) and
  weighted-rating-scale (RS/RT/RW/RK) customs that `metric_type.py` labels
  "Custom Metric" are deliberately NOT eligible — the check reads raw platform
  signals only.
* `bounded_scale_kind` / `is_platform_metric` — "what shape is the answer, and
  does the platform band it?" This is the reporting-capability question that
  `is_filterable`, `is_segmentable` and the dashboard-capability layer share.
  It is deliberately wider than journey eligibility (an unflagged rating scale
  still answers on a bounded scale) and wider than `metric_type`'s Custom
  Metric label (which requires option weights, because weights are what make a
  scale *scoreable* — they are not what makes it *facetable*).

Keep `bounded_scale_kind` in step with
`taggers/question/dashboard_capability.py::_response_format`: the two must not
disagree about whether a question's answer lands on a scale, or `is_filterable`
and `crosstab_axis_role` will state opposite things about the same question.
"""

from __future__ import annotations

from models import evidence as ev
from models.survey import QuestionContext

# rs_type → (metric name, what the platform flag means)
_RS_TYPE_METRICS = {
    2: ("NPS/eNPS", "the platform flags this question as a Net Promoter Score (NPS) item"),
    3: ("CES", "the platform flags this question as a Customer Effort Score (CES) item"),
    4: ("CSAT", "the platform flags this question as a Customer Satisfaction (CSAT) item"),
}

# Question types whose answer is a point on a bounded scale even when the
# platform has not flagged the question as a metric. Weights are NOT required:
# an unweighted 1-5 rating still has five discrete, ordered answers.
_SCALE_TYPES = {"RS", "RT", "RW", "RK"}
_MATRIX_TYPES = {"GR", "GC", "RG", "GQ"}

# bounded_scale_kind() → the phrase a tagger drops into its evidence sentence.
_SCALE_KIND_DETAIL = {
    "nps": "an NPS/eNPS item (rs_type=2), which the platform bands into "
           "Promoter / Passive / Detractor",
    "ces": "a Customer Effort Score item (rs_type=3), which the platform bands "
           "into low / medium / high effort",
    "csat": "a CSAT item (rs_type=4), which the platform bands into satisfied / "
            "neutral / dissatisfied",
    "custom_metric": "a customer-defined metric the platform scores and bands "
                     "like a standard one",
    "rating_scale": "a rating or ranking scale, so every answer is one of a "
                    "fixed, ordered set of scale points",
    "matrix_row": "one row of a matrix/grid, so its answer is one point on the "
                  "grid's shared scale",
}


def is_platform_metric(question: QuestionContext) -> bool:
    """True when the PLATFORM scores this question as a metric.

    Narrower than `bounded_scale_kind`, and the distinction earns its keep: a
    platform metric comes with canonical bands (Promoter / Passive / Detractor,
    satisfied / neutral / dissatisfied), which is what makes it usable as a
    grouping variable rather than only as a filter facet. An unflagged rating
    scale has scale points but no agreed banding.
    """
    return question.rs_type in _RS_TYPE_METRICS or bool(question.is_custom_metric)


def bounded_scale_kind(question: QuestionContext) -> str | None:
    """Which kind of bounded scale this question's answer lands on, or None.

    One of `nps` / `ces` / `csat` / `custom_metric` / `rating_scale` /
    `matrix_row`. Every one of them yields a discrete, bounded answer set, which
    is the property a report filter needs — so this is the predicate behind
    "a metric is filterable too", not just "a choice list is".

    The platform's own flags win over the question type, exactly as in
    `metric_type` and `_response_format`: a CSAT delivered as a radio button is
    a CSAT first.
    """
    if question.rs_type in _RS_TYPE_METRICS:
        return {2: "nps", 3: "ces", 4: "csat"}[question.rs_type]
    if question.is_custom_metric:
        return "custom_metric"
    if question.question_type in _SCALE_TYPES:
        return "rating_scale"
    if question.question_type in _MATRIX_TYPES:
        return "matrix_row"
    return None


def describe_scale_kind(kind: str) -> str:
    """The evidence phrase for a `bounded_scale_kind` result."""
    return _SCALE_KIND_DETAIL.get(kind, "a scored question")


def is_journey_eligible_metric(question: QuestionContext) -> tuple[bool, dict]:
    """Return (eligible, typed evidence explaining the verdict).

    Eligible iff one of:
      - rs_type == 2  (NPS / eNPS)
      - rs_type == 3  (CES)
      - rs_type == 4  (CSAT)
      - is_custom_metric is True  (platform-flagged custom metric)

    The evidence is a `models.evidence` dict rather than a sentence because it
    is handed straight to `TagResult.evidence` by the journey taggers — the
    reason a question did NOT get a journey stage is the single most-asked
    question about this pipeline's output.
    """
    rs = question.rs_type
    if rs in _RS_TYPE_METRICS:
        metric, why = _RS_TYPE_METRICS[rs]
        return True, ev.rule(
            "question.journey.eligible_rs_type",
            f"Journey-eligible: {why} (rs_type={rs}), and journey stages are only "
            "assigned to metric questions.",
            stage=5,
            inputs={"rs_type": rs, "metric": metric},
        )
    if question.is_custom_metric:
        return True, ev.rule(
            "question.journey.eligible_custom_metric",
            "Journey-eligible: the platform flags this question as a custom metric, "
            "which counts alongside NPS, CES and CSAT.",
            stage=5,
            inputs={"is_custom_metric": True, "rs_type": rs},
        )
    return False, ev.rule(
        "question.journey.not_a_metric",
        "Not journey-eligible. A journey stage is only assigned to metric questions — "
        "NPS (rs_type 2), CES (3), CSAT (4), or a platform-flagged custom metric. This "
        f"question has rs_type={rs} and no custom-metric flag, so it is skipped rather "
        "than guessed at. Note that a matrix or weighted-rating question that "
        "`metric_type` labels a Custom Metric still does not qualify here — this check "
        "reads only raw platform signals.",
        stage=5,
        inputs={"rs_type": rs, "is_custom_metric": False,
                "question_type": question.question_type},
    )
