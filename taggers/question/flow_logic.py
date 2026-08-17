"""Flow logic role tagger: identifies branching, piping, and routing roles.

The structural half of a hybrid dimension. Everything here is read off a platform
flag — `questionType == "HR"`, `isFollowupQuestion`, `metricQuestion`, piping
markers in the title — because `survey_structure.json` carries no skip-logic,
display-condition, quota or loop definitions at all. That is also why the empty
list is the overwhelmingly common answer, and why `Skip Logic Source`,
`Loop Trigger` and `Quota Controller` never appear from this pass.

LLM Call 2 adds the first two of those (plus wording-evident Branching and
Termination triggers) at a much lower confidence, unioned on by
`pipeline/llm_enhance._apply_flow_logic_inference`. It can only ADD: these entries
are facts and an inference must not be able to delete one.
"""

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowLogicTagger(QuestionTagger):
    name = "question.flow_logic"
    tag_dimension = "flow_logic_role"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=[], source="deterministic", status="skipped",
                             evidence=ev.content_message("flow_logic_role", stage=3))

        roles: list[str] = []
        # Multi-label: record which signal put each role on the list, since one
        # sentence about the whole list would explain none of the entries.
        why: list[dict] = []

        # Branching Trigger: hidden radio (routing question)
        if question.question_type == "HR":
            roles.append("Branching Trigger")
            why.append(ev.component("Branching Trigger",
                                    "hidden radio (type HR) — a routing question the "
                                    "respondent never sees"))

        # Branching Target: follow-up question (conditionally shown)
        if question.is_followup_question:
            roles.append("Branching Target")
            why.append(ev.component("Branching Target",
                                    "flagged as a follow-up, so it is shown "
                                    "conditionally on an earlier answer"))

        # Piping Target: question contains piping markers
        if question.has_piping_markers:
            roles.append("Piping Target")
            why.append(ev.component("Piping Target",
                                    "question text contains piping markers"))

        # Piping Source: this question's ID is referenced as metricQuestion by another
        is_referenced = any(
            q.metric_question_id == question.question_id
            for q in context.questions
            if q.is_followup_question
        )
        if is_referenced:
            roles.append("Piping Source")
            why.append(ev.component("Piping Source",
                                    "another follow-up question names this one as its "
                                    "metric question"))

        # Termination Trigger: Yes/No radio at position ≤1
        if (
            question.question_type == "R"
            and question.effective_position_ratio <= 0.1
            and len(question.answer_options) == 2
        ):
            opt_texts = {o.answer_text.lower().strip() for o in question.answer_options}
            if opt_texts == {"yes", "no"}:
                roles.append("Termination Trigger")
                why.append(ev.component("Termination Trigger",
                                        "Yes/No radio in the opening 10% of the "
                                        "survey — the shape of a screener"))

        if not roles:
            return TagResult(
                value=roles,
                source="deterministic",
                confidence=1.0,
                evidence=ev.rule(
                    "question.flow_logic_role.no_logic",
                    "No routing role is visible in the platform's flags: not a hidden "
                    "radio, not a follow-up, no piping markers, not piped from, not an "
                    "opening Yes/No screener. An empty list here is a finding, not a "
                    "missing value — though it is a finding about the FLAGS, since the "
                    "payload holds no logic definitions to check. LLM Call 2 may still "
                    "add a role it reads in the wording.",
                    stage=3,
                    inputs={"question_type": question.question_type},
                ),
            )

        return TagResult(
            value=roles,
            source="deterministic",
            confidence=0.90,
            evidence=ev.hybrid(
                "question.flow_logic_role.detected",
                f"{len(roles)} routing role(s) detected from the question's structure; "
                "each is listed below with the signal that produced it.",
                components=why,
                stage=3,
                inputs={"question_type": question.question_type},
            ),
        )


def create_tagger() -> FlowLogicTagger:
    return FlowLogicTagger()
