"""Flow respondent experience tagger: LLM-based with structural priors."""

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowExperienceTagger(QuestionTagger):
    name = "question.flow_experience"
    tag_dimension = "flow_respondent_experience"
    stage = 5
    depends_on = ["question.flow_placement"]
    source_type = "llm"
    # This tagger TAGS content messages rather than skipping them, so the
    # base class must not short-circuit them away before it is called.
    skips_content_messages = False

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            # CM at the start = Trust Builder (welcome/instructions)
            if question.position_index <= 1:
                title_lower = question.title.lower()
                if any(kw in title_lower for kw in ["welcome", "thank you", "this survey"]):
                    return TagResult(
                        value="Trust Builder",
                        source="hybrid",
                        confidence=0.85,
                        evidence=ev.rule(
                            "question.flow_experience.welcome_message",
                            "A content message in the first two positions whose text "
                            "reads as a welcome or introduction. This is what sets "
                            "expectations before anyone is asked anything — note this "
                            "is one of the few dimensions a content message DOES get.",
                            stage=5,
                            inputs={"position_index": question.position_index,
                                    "is_content_message": True},
                            quote=question.title,
                        ),
                    )
            # Mid-survey CM = section header = Progress Anchor
            if question.position_index > 1:
                return TagResult(
                    value="Progress Anchor",
                    source="hybrid",
                    confidence=0.80,
                    evidence=ev.rule(
                        "question.flow_experience.section_header",
                        "A content message partway through the survey — a section "
                        "header. It tells the respondent where they are and how much "
                        "is left, which is what keeps completion rates up.",
                        stage=5,
                        inputs={"position_index": question.position_index,
                                "is_content_message": True},
                        quote=question.title,
                    ),
                )
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.flow_experience.opening_cm_not_welcome",
                    "A content message at the very start whose text does not read as a "
                    "welcome or introduction, so it plays no identifiable role in the "
                    "respondent's experience.",
                    stage=5,
                    inputs={"position_index": question.position_index,
                            "is_content_message": True},
                ),
            )

        placement = accumulator.get_question_tag_value(question.question_id, "flow_placement")

        # Opening question = Trust Builder (easy entry point)
        if placement == "Opening":
            return TagResult(
                value="Trust Builder",
                source="hybrid",
                confidence=0.75,
                evidence=ev.rule(
                    "question.flow_experience.opening_question",
                    "flow_placement already put this first in the survey. The opening "
                    "question is where a respondent decides whether to continue, so it "
                    "functions as the trust builder whatever it asks.",
                    stage=5,
                    inputs={"flow_placement": "Opening"},
                ),
            )

        # Large matrix group = Effort Checkpoint
        if question.matrix_group_size > 6:
            return TagResult(
                value="Effort Checkpoint",
                source="hybrid",
                confidence=0.80,
                evidence=ev.statistic(
                    "question.flow_experience.large_matrix",
                    f"A {question.matrix_group_size}-row grid. Past about six rows this "
                    "is the point in the survey where effort spikes and people start "
                    "abandoning or straight-lining.",
                    measure="matrix_group_size",
                    observed=question.matrix_group_size,
                    threshold=6,
                    stage=5,
                    inputs={"flow_placement": placement or "(unset)"},
                ),
            )

        # NPS/CSAT near end = Progress Anchor
        if (question.is_nps or question.is_csat) and placement in ("Deep Dive", "Closing"):
            return TagResult(
                value="Progress Anchor",
                source="hybrid",
                confidence=0.80,
                evidence=ev.hybrid(
                    "question.flow_experience.late_headline_metric",
                    f"A headline metric placed in the {placement} section. Reaching the "
                    "big question signals to the respondent that the survey is nearly "
                    "over, which is what a progress anchor does.",
                    components=[
                        ev.component("NPS" if question.is_nps else "CSAT",
                                     "headline metric question"),
                        ev.component(f"flow_placement={placement}",
                                     "late in the survey"),
                    ],
                    stage=5,
                ),
            )

        # Open-ended after rating block = Re-engagement Point
        if question.question_type == "T" and not question.is_followup_question:
            # Check if previous questions were all rating types
            idx = question.position_index
            if idx >= 3:
                prev_types = [
                    q.question_type for q in context.questions[max(0, idx - 3):idx]
                    if not q.is_content_message
                ]
                if all(t in ("RS", "RW", "GR", "RG", "RT") for t in prev_types if t):
                    return TagResult(
                        value="Re-engagement Point",
                        source="hybrid",
                        confidence=0.70,
                        evidence=ev.rule(
                            "question.flow_experience.text_after_ratings",
                            "An open text box immediately following a run of rating "
                            "questions. After several scales in a row, being asked to "
                            "say something in your own words re-engages attention.",
                            stage=5,
                            inputs={"preceding_question_types": prev_types,
                                    "flow_placement": placement or "(unset)"},
                        ),
                    )

        # Default — requires LLM
        return TagResult(
            value="Progress Anchor",
            source="llm",
            confidence=0.40,
            evidence=ev.fallback(
                "question.flow_experience.deferred_to_llm",
                f"No structural rule fired: flow_placement is "
                f"{placement or 'unset'}, the question is not in a large grid, not a "
                "late headline metric, and not an open-end following a rating block. "
                "What a question does to the respondent's experience needs reading the "
                "wording, so this 0.40 placeholder is for LLM Call 2 to replace.",
                stage=5,
                inputs={"flow_placement": placement or "(unset)",
                        "question_type": question.question_type},
            ),
        )


def create_tagger() -> FlowExperienceTagger:
    return FlowExperienceTagger()
