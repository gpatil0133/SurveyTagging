"""Flow dependency tagger: identifies dependency relationships between questions."""

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowDependencyTagger(QuestionTagger):
    name = "question.flow_dependency"
    tag_dimension = "flow_dependency"
    stage = 3
    source_type = "deterministic"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        # Conditionally Shown: follow-up question
        if question.is_followup_question:
            return TagResult(
                value="Conditionally Shown",
                source="deterministic",
                confidence=0.95,
                evidence=ev.rule(
                    "question.flow_dependency.followup",
                    f"The platform marks this as a follow-up to question "
                    f"{question.metric_question_id}, so respondents only reach it "
                    "depending on how they answered that one. Its response base is "
                    "therefore a subset of the survey's.",
                    stage=3,
                    inputs={"parent_question_id": question.metric_question_id},
                ),
            )

        # References Prior: has piping markers
        if question.has_piping_markers:
            return TagResult(
                value="References Prior",
                source="deterministic",
                confidence=0.90,
                evidence=ev.rule(
                    "question.flow_dependency.piping",
                    "The question text contains piping markers, so its wording is "
                    "assembled at runtime from an earlier answer. It reads differently "
                    "for different respondents.",
                    stage=3,
                    inputs={"piping_markers": question.piping_markers[:2]},
                    quote=question.title,
                ),
            )

        # Mutually Exclusive Block: part of matrix group with >1 member
        if question.matrix_group_size > 1 and question.matrix_group_title:
            return TagResult(
                value="Mutually Exclusive Block",
                source="deterministic",
                confidence=0.85,
                evidence=ev.rule(
                    "question.flow_dependency.matrix_group",
                    f"One row of a {question.matrix_group_size}-row matrix presented "
                    "under a single stem. It is answered alongside its siblings and "
                    "should be read with them, not on its own.",
                    stage=3,
                    inputs={"matrix_group_title": question.matrix_group_title,
                            "matrix_group_size": question.matrix_group_size},
                ),
            )

        # Carries Forward: this question is referenced by follow-ups
        is_referenced = any(
            q.metric_question_id == question.question_id
            for q in context.questions
            if q.is_followup_question
        )
        if is_referenced:
            return TagResult(
                value="Carries Forward",
                source="deterministic",
                confidence=0.90,
                evidence=ev.rule(
                    "question.flow_dependency.referenced_by_followup",
                    "This question is not itself conditional, but at least one "
                    "follow-up question names it as the metric it branches on. Its "
                    "answers drive who sees what later in the survey.",
                    stage=3,
                    inputs={"question_id": question.question_id},
                ),
            )

        # Default: Independent
        return TagResult(
            value="Independent",
            source="deterministic",
            confidence=1.0,
            evidence=ev.rule(
                "question.flow_dependency.independent",
                "The question stands alone: not conditionally shown, no piping in its "
                "text, not part of a matrix group, and no follow-up branches off it. "
                "Every respondent who reaches the survey sees it as written.",
                stage=3,
            ),
        )


def create_tagger() -> FlowDependencyTagger:
    return FlowDependencyTagger()
