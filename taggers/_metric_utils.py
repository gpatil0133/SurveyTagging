"""Shared helpers for journey-eligibility across question-level taggers.

A question is "journey-eligible" if it is one of the four kinds we track in
the Custom Journey: NPS, CSAT, CES (identified by rs_type), or a platform-
flagged custom metric (is_custom_metric=True).

Heuristic matrix-type (GR/GC/RG/GQ) and weighted-rating-scale (RS/RT/RW/RK)
customs that `metric_type.py` labels "Custom Metric" are deliberately NOT
eligible — the check reads raw platform signals from QuestionContext only.
"""

from __future__ import annotations

from models.survey import QuestionContext


def is_journey_eligible_metric(question: QuestionContext) -> tuple[bool, str]:
    """Return (eligible, evidence).

    Eligible iff one of:
      - rs_type == 2  (NPS / eNPS)
      - rs_type == 3  (CES)
      - rs_type == 4  (CSAT)
      - is_custom_metric is True  (platform-flagged custom metric)
    """
    rs = question.rs_type
    if rs == 2:
        return True, "rs_type=2 (NPS/eNPS)"
    if rs == 3:
        return True, "rs_type=3 (CES)"
    if rs == 4:
        return True, "rs_type=4 (CSAT)"
    if question.is_custom_metric:
        return True, "is_custom_metric=True (platform-flagged custom metric)"
    return False, "not a journey-eligible metric question (needs NPS/CSAT/CES/custom metric)"
