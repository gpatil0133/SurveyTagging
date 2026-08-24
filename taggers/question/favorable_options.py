"""favorable_options tagger — which answer options count as favorable,
unfavorable and neutral (V8, Phase 4).

Stage 3, hybrid, user_defined. No tag dependencies.

**What it fills.** `metricDetails[].positiveAnswerOptionIds` and its negative and
neutral siblings on the widget payload. Without it, Percent Favorable and Net
Intent are unbuildable on every question that is not rs_type 2, 3 or 4 — which
is most of the weighted scales in a real survey.

**Shape.** `{"positive": [answer_id...], "negative": [...], "neutral": [...]}`
rather than an enum. The values are references into the question's own answer
options, exactly like `driver_link` and `verbatim_link` are references to
question ids — hence `user_defined`, so nothing enum-checks them.

This is the first dimension in the taxonomy to say anything about ANSWER
OPTIONS. Everything else stops at the question.

**Derivation.** Three tiers, strongest first:

    rs_type 2      the fixed NPS bands: 9-10 promoters, 7-8 passives, 0-6
                   detractors. Not inferred — this is the metric's definition.
    weighted scale split at the midpoint of the weight range; above is positive,
                   below negative, exactly the midpoint neutral.
    unweighted     handed to LLM Call 2, which reads the option TEXT. This is
                   the one case a rule genuinely cannot decide: an unweighted
                   list carries no direction, and "Very satisfied" vs "Strongly
                   disagree" is a reading of language, not of structure.

Known limitation, recorded rather than guessed at: a reverse-coded weighted
scale (a CES where 1 means "very easy") comes out inverted, because the weights
say which end is high and nothing in the payload says which end is good. The LLM
tier does not have this problem; the weighted tier is faster and covers far more
questions, so it stays first. A reverse-coded scale is visible in the evidence's
`weight_range` if a reviewer wants to spot them.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# The NPS bands, by score. Fixed by the metric's own definition.
_NPS_POSITIVE = {9, 10}
_NPS_NEUTRAL = {7, 8}

# Answer shapes with a direction worth splitting at all. A checkbox list, a
# routing question and a demographic have favorable options only in the sense
# that any option is — the concept does not apply.
_DIRECTIONAL_TYPES = {"RS", "RT", "RW", "GR", "GC", "RG", "GQ", "L", "R", "SR"}


def _empty() -> dict[str, list[int]]:
    return {"positive": [], "negative": [], "neutral": []}


def _nps_split(q: QuestionContext) -> dict[str, list[int]]:
    """Promoters / passives / detractors, keyed off the option's own score.

    The score is the weight when the platform supplied one and the option's
    position otherwise — an 11-point NPS scale is emitted in order, so position
    IS the score when weights are absent.

    Weights are shifted so the LOWEST is zero. Some surveys weight the 0-10
    scale 1-11, and reading those literally puts "11" outside the promoter band
    and slides every boundary down one — the whole metric off by one point,
    silently. The shift makes both conventions land on the same bands, and it
    is safe because NPS always carries the full scale.
    """
    weights = [o.weight for o in q.answer_options if o.weight is not None]
    offset = min(weights) if weights else 0
    out = _empty()
    for index, opt in enumerate(q.answer_options):
        score = int(opt.weight - offset) if opt.weight is not None else index
        if score in _NPS_POSITIVE:
            out["positive"].append(opt.answer_id)
        elif score in _NPS_NEUTRAL:
            out["neutral"].append(opt.answer_id)
        else:
            out["negative"].append(opt.answer_id)
    return out


def _midpoint_split(q: QuestionContext) -> tuple[dict[str, list[int]], float, float, float]:
    """Split a weighted scale at the midpoint of its weight range."""
    weights = [o.weight for o in q.answer_options if o.weight is not None]
    lo, hi = min(weights), max(weights)
    mid = (lo + hi) / 2
    out = _empty()
    for opt in q.answer_options:
        if opt.weight is None:
            # An unweighted option inside a weighted scale is an escape hatch —
            # "Not applicable", "Prefer not to say". It is neither favorable nor
            # unfavorable, and counting it either way skews the metric.
            out["neutral"].append(opt.answer_id)
        elif opt.weight > mid:
            out["positive"].append(opt.answer_id)
        elif opt.weight < mid:
            out["negative"].append(opt.answer_id)
        else:
            out["neutral"].append(opt.answer_id)
    return out, lo, hi, mid


class FavorableOptionsTagger(QuestionTagger):
    name = "question.favorable_options"
    tag_dimension = "favorable_options"
    stage = 3
    source_type = "hybrid"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("favorable_options", stage=3))

        if not q.answer_options:
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.favorable_options.no_options",
                    "The question has no answer options — free text, a contact block or "
                    "a file upload — so there is nothing to divide into favorable and "
                    "unfavorable.",
                    stage=3,
                    inputs={"question_type": q.question_type, "option_count": 0},
                ),
            )

        # Tier 1: NPS bands, by definition rather than by inference.
        if q.rs_type == 2:
            value = _nps_split(q)
            return TagResult(
                value=value, source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.favorable_options.nps_bands",
                    "An NPS item (rs_type=2). The bands are part of the metric's "
                    "definition, not a reading of this survey: 9-10 promoters, 7-8 "
                    "passives, 0-6 detractors. Splitting it any other way would stop it "
                    "being NPS.",
                    stage=3,
                    inputs={"rs_type": 2,
                            "positive": len(value["positive"]),
                            "neutral": len(value["neutral"]),
                            "negative": len(value["negative"])},
                ),
            )

        has_weights = any(o.weight is not None for o in q.answer_options)

        # Tier 2: a weighted scale carries its own direction.
        if has_weights:
            value, lo, hi, mid = _midpoint_split(q)
            if not value["positive"] and not value["negative"]:
                # Every option shares one weight — a flat scale has no direction.
                return TagResult(
                    value=None, source="deterministic", status="skipped",
                    evidence=ev.rule(
                        "question.favorable_options.flat_weights",
                        f"Every option carries the same weight ({lo}), so the scale has "
                        "no direction to split on. Weighted in name only.",
                        stage=3,
                        inputs={"weight_range": [lo, hi]},
                    ),
                )
            return TagResult(
                value=value, source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.favorable_options.weight_midpoint",
                    f"The options carry weights from {lo} to {hi}, so the scale states "
                    f"its own direction. Split at the midpoint ({mid}): above is "
                    "favorable, below unfavorable, exactly the midpoint neutral. "
                    "0.90 rather than 1.0 because the weights say which end is HIGH, "
                    "not which end is GOOD — a reverse-coded scale would come out "
                    "inverted.",
                    stage=3,
                    inputs={"weight_range": [lo, hi], "midpoint": mid,
                            "positive": len(value["positive"]),
                            "neutral": len(value["neutral"]),
                            "negative": len(value["negative"])},
                ),
            )

        # Tier 3: unweighted. Only the option text carries the direction, so
        # this is LLM Call 2's — and only for shapes where the concept applies.
        if q.question_type in _DIRECTIONAL_TYPES and 2 <= len(q.answer_options) <= 15:
            return TagResult(
                value=None, source="hybrid", status="pending_llm", confidence=0.0,
                evidence=ev.rule(
                    "question.favorable_options.needs_option_text",
                    f"An unweighted list of {len(q.answer_options)} options. Nothing "
                    "structural says which end is favorable — an unweighted list carries "
                    "no direction at all — so the decision needs the option WORDING "
                    '("Very satisfied", "Strongly disagree"), which is LLM Call 2\'s to '
                    "read.",
                    stage=3,
                    inputs={"question_type": q.question_type,
                            "option_count": len(q.answer_options),
                            "has_weights": False},
                ),
            )

        return TagResult(
            value=None, source="deterministic", status="skipped",
            evidence=ev.rule(
                "question.favorable_options.not_directional",
                f"Type {q.question_type} with {len(q.answer_options)} option(s): a "
                "checkbox list, a routing question or a plain category set. Its options "
                "are alternatives rather than points on a good-to-bad scale, so "
                "favorability does not apply — Percent Favorable and Net Intent are not "
                "metrics this question can carry.",
                stage=3,
                inputs={"question_type": q.question_type,
                        "option_count": len(q.answer_options),
                        "has_weights": False},
            ),
        )


def create_tagger() -> FavorableOptionsTagger:
    return FavorableOptionsTagger()
