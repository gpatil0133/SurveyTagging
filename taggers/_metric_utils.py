"""Shared helpers for journey-eligibility across question-level taggers.

A question is "journey-eligible" if it is one of the four kinds we track in
the Custom Journey: NPS, CSAT, CES (identified by rs_type), or a platform-
flagged custom metric (is_custom_metric=True).

Heuristic matrix-type (GR/GC/RG/GQ) and weighted-rating-scale (RS/RT/RW/RK)
customs that `metric_type.py` labels "Custom Metric" are deliberately NOT
eligible — the check reads raw platform signals from QuestionContext only.
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
