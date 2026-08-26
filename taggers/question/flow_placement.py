"""Flow placement tagger: position-based classification within survey flow."""

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowPlacementTagger(QuestionTagger):
    name = "question.flow_placement"
    tag_dimension = "flow_placement"
    stage = 3
    source_type = "deterministic"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        total_non_cm = len(context.non_cm_questions)

        if total_non_cm == 0:
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.flow_placement.no_real_questions",
                    "The survey contains no non-content-message questions at all, so "
                    "there is no flow to place anything within.",
                    stage=3,
                    inputs={"non_cm_question_count": 0},
                ),
            )

        if total_non_cm == 1:
            return TagResult(
                value="Standalone", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.flow_placement.only_question",
                    "This is the survey's only real question — there is no opening, "
                    "middle or close for it to sit in.",
                    stage=3,
                    inputs={"non_cm_question_count": 1},
                ),
            )

        ratio = question.effective_position_ratio

        def _placement(rule_suffix: str, detail: str, threshold=None) -> dict:
            return ev.statistic(
                f"question.flow_placement.{rule_suffix}",
                detail,
                measure="position_ratio",
                observed=round(ratio, 2),
                threshold=threshold,
                stage=3,
                inputs={"non_cm_question_count": total_non_cm},
            )

        if ratio == 0.0:
            return TagResult(
                value="Opening", source="deterministic", confidence=1.0,
                evidence=_placement(
                    "first_question",
                    "This is the first real question in the survey — position ratio 0.",
                ),
            )

        if ratio <= 0.15:
            return TagResult(
                value="Warm-up", source="deterministic", confidence=0.90,
                evidence=_placement(
                    "opening_band",
                    f"It sits {ratio:.0%} of the way through the survey, inside the "
                    "opening 15% where easy warm-up questions go.",
                    threshold=0.15),
            )

        if ratio <= 0.75:
            return TagResult(
                value="Core Block", source="deterministic", confidence=0.90,
                evidence=_placement(
                    "core_band",
                    f"It sits {ratio:.0%} of the way through — past the warm-up and "
                    "before the final quarter, which is where the substantive "
                    "questions live.",
                    threshold=0.75),
            )

        if ratio <= 0.90:
            return TagResult(
                value="Deep Dive", source="deterministic", confidence=0.85,
                evidence=_placement(
                    "deep_dive_band",
                    f"It sits {ratio:.0%} of the way through, in the 75-90% band where "
                    "follow-up and detail questions are typically placed.",
                    threshold=0.90),
            )

        return TagResult(
            value="Closing", source="deterministic", confidence=0.90,
            evidence=_placement(
                "closing_band",
                f"It sits {ratio:.0%} of the way through — the final 10% of the "
                "survey, where demographics and closing questions go.",
                threshold=0.90),
        )


def create_tagger() -> FlowPlacementTagger:
    return FlowPlacementTagger()
