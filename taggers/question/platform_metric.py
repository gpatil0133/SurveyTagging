"""platform_metric tagger — the ordered list of platform metric codes valid for
this question, primary first (V8, Phase 2).

Stage 4, multi-label, deterministic. Depends on response_format (Stage 3).

`metricType` is required on every widget insert and, before V8, nothing in the
taxonomy could produce one: nine analytical values against the platform's twenty
codes, with the single most-used code — Weighted Score, the default for every
grid and rating question — absent entirely.

This dimension maps one-to-one onto the payload's `metricDetails[]` array, where
POSITION becomes `orderId`. A table on an NPS question emits
`[NPS, Group Percentage]` and gets both rows, exactly as the manual UI does; a
chart takes the head of the list.

Why a second dimension rather than re-keying `calculation_type`: coupling that
one to a single vendor's enum would break its other consumers (Reporting and S5
have no business knowing RMX codes). The dependency is strictly one-way instead
— this derives from the raw signals, and `calculation_type` is a coarsening of
the same computation. Both call `taggers/_platform_metrics.py`, so they cannot
restate each other and drift, which is the failure that killed `control_role` in
V7.3.

Deterministic, never LLM-refined.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers._platform_metrics import (
    METRIC_CODES,
    compute_platform_metrics,
    metric_from_platform_hint,
)
from taggers.base import QuestionTagger


class PlatformMetricTagger(QuestionTagger):
    name = "question.platform_metric"
    tag_dimension = "platform_metric"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.response_format"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=[], source="deterministic", status="skipped",
                             evidence=ev.content_message("platform_metric", stage=4))

        fmt = accumulator.get_question_tag_value(q.question_id, "response_format")
        if not fmt:
            from taggers.question.dashboard_capability import _response_format
            fmt = _response_format(q)

        metrics = compute_platform_metrics(q, fmt)

        if not metrics:
            # An empty list, not a skip: "the platform offers no metric for this
            # answer shape" is a finding a composer needs, and it is different
            # from "never evaluated".
            return TagResult(
                value=[], source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.platform_metric.no_metric_offered",
                    f"A {fmt} answer is prose, an identifier or an attachment. The "
                    "platform offers no metric for it, so a widget built on this "
                    "question can only be a table — there is no metricType to send.",
                    stage=4,
                    inputs={"response_format": fmt,
                            "question_type": q.question_type},
                ),
            )

        hint = metric_from_platform_hint(q.calculation_type)
        primary = metrics[0]
        codes = [METRIC_CODES.get(m, "??") for m in metrics]

        detail = (
            f"A {fmt} answer supports {len(metrics)} platform metric(s), primary first: "
            + ", ".join(f"{m} ({METRIC_CODES.get(m, '??')})" for m in metrics)
            + ". Position is meaningful — it becomes `orderId` in the payload's "
            "metricDetails[] array, so a table on this question renders these rows in "
            "this order and a chart uses the first."
        )
        if hint and hint == primary:
            detail += (" The primary is the metric the survey author configured in the "
                       "platform itself, which outranks every shape rule.")

        return TagResult(
            value=metrics, source="deterministic",
            confidence=1.0 if (hint or q.rs_type in (2, 3, 4) or q.is_custom_metric)
            else 0.90,
            evidence=ev.rule(
                "question.platform_metric.derived",
                detail,
                stage=4,
                inputs={"response_format": fmt,
                        "rs_type": q.rs_type,
                        "is_custom_metric": q.is_custom_metric,
                        "platform_calculation_type": q.calculation_type or None,
                        "metric_codes": codes},
            ),
        )


def create_tagger() -> PlatformMetricTagger:
    return PlatformMetricTagger()
