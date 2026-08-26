"""calculation_type tagger — how the question's responses should be aggregated.

Stage 4, deterministic. Depends on response_format (Stage 3) and metric_name
(Stage 3, for the evidence sentence only).

V8 made this a *coarsening* of `platform_metric` rather than a parallel cascade.
The two dimensions answer the same question at different altitudes — this one is
the vendor-neutral analytical statement Reporting and S5 read, `platform_metric`
is the RMX code list a widget payload needs — and stating that twice is how
`control_role` drifted out of agreement with `is_filterable` before V7.3 removed
it. The derivation now lives once, in `taggers/_platform_metrics.py`, and both
taggers call it, so they cannot disagree by construction.

Three corrections came with that move:

* The platform's own `calculationType` hint no longer normalizes "weighted" to
  `Mean`. The author explicitly configured a weighted calculation and we were
  discarding it — while RMX's own default for that question is Weighted Score.
* CSAT and CES stop resolving to `Mean`. The platform has dedicated metrics for
  both and defaults to them; the honest coarse word is `Weighted Score`.
* Ranking questions keep `Mean Rank` here, because that IS what a report says
  about them — but the platform target behind it moved to Overall Rank on tables
  and Weighted Score on graphs, which is `platform_metric`'s job to carry.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers._platform_metrics import (
    METRIC_CODES,
    coarsen_to_calculation_type,
    compute_platform_metrics,
    metric_from_platform_hint,
)
from taggers.base import QuestionTagger

# How confident the coarsening is, by where the primary metric came from. An
# author's explicit setting is a fact; a platform metric flag is a fact; a shape
# rule is an inference.
_CONF_AUTHOR = 1.0
_CONF_PLATFORM_FLAG = 1.0
_CONF_SHAPE = 0.90
_CONF_NONE = 0.70


class CalculationTypeTagger(QuestionTagger):
    name = "question.calculation_type"
    tag_dimension = "calculation_type"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.response_format", "question.metric_name"]

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        fmt = accumulator.get_question_tag_value(q.question_id, "response_format")
        if not fmt:
            # response_format is Stage 3 and never skips a real question, so this
            # only happens if that tagger failed. Recompute rather than guess.
            from taggers.question.dashboard_capability import _response_format
            fmt = _response_format(q)

        metric_name = accumulator.get_question_tag_value(q.question_id, "metric_name")
        metrics = compute_platform_metrics(q, fmt)
        value = coarsen_to_calculation_type(metrics, fmt, len(q.answer_options))

        if not metrics:
            return TagResult(
                value="Not Applicable", source="deterministic", confidence=_CONF_NONE,
                evidence=ev.fallback(
                    "question.calculation_type.nothing_to_aggregate",
                    f"Nothing to aggregate: a {fmt} answer is prose, an identifier or an "
                    "attachment, so the platform offers no metric for it and there is no "
                    "aggregation to name. Typically a free-text box, a contact block, a "
                    "file upload or an unrecognized question type.",
                    stage=4,
                    inputs={"response_format": fmt,
                            "question_type": q.question_type,
                            "metric_name": metric_name or "(unset)"},
                ),
            )

        primary = metrics[0]
        hint = metric_from_platform_hint(q.calculation_type)

        if hint and hint == primary:
            return TagResult(
                value=value, source="deterministic", confidence=_CONF_AUTHOR,
                evidence=ev.rule(
                    "question.calculation_type.platform_declared",
                    f"The survey author configured this question's calculation in the "
                    f"platform itself. That setting names the {primary} metric, which "
                    f"coarsens to {value}. An explicit author choice outranks every "
                    "shape rule — including a weighted calculation, which this pipeline "
                    "used to discard by rewriting it as a mean.",
                    stage=4,
                    inputs={"platform_calculation_type": q.calculation_type,
                            "platform_metric": primary,
                            "metric_code": METRIC_CODES.get(primary),
                            "response_format": fmt},
                ),
            )

        platform_flagged = q.rs_type in (2, 3, 4) or bool(q.is_custom_metric)
        return TagResult(
            value=value, source="deterministic",
            confidence=_CONF_PLATFORM_FLAG if platform_flagged else _CONF_SHAPE,
            evidence=ev.rule(
                "question.calculation_type.coarsened_platform_metric",
                f"A {fmt} answer. The platform's primary metric for it is {primary}"
                + (f" (rs_type={q.rs_type}, a flag rather than an inference)"
                   if platform_flagged else "")
                + f", and the vendor-neutral name for that aggregation is {value}. "
                "This dimension is a coarsening of platform_metric rather than a second "
                "opinion, so the two cannot state different things about one question.",
                stage=4,
                inputs={"response_format": fmt,
                        "platform_metric": primary,
                        "metric_code": METRIC_CODES.get(primary),
                        "metric_name": metric_name or "(unset)",
                        "option_count": len(q.answer_options)},
            ),
        )


def create_tagger() -> CalculationTypeTagger:
    return CalculationTypeTagger()
