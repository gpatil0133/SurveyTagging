"""Relationship type tagger: LLM-based with structural priors."""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


# Parallel.ai agent CX taxonomy → our taxonomy. Agents emit Transactional /
# Subscription / Membership / Contractual / Partnership / Project-Based.
# Our `relationship_type` taxonomy is Transactional / Relational / Journey-based / Pulse.
_AGENT_RELATIONSHIP_MAP: dict[str, str] = {
    "Transactional": "Transactional",
    "Subscription": "Relational",
    "Membership": "Relational",
    "Contractual": "Relational",
    "Partnership": "Relational",
    "Project-Based": "Transactional",
}


class RelationshipTagger(ProjectTagger):
    name = "project.relationship"
    tag_dimension = "relationship_type"
    stage = 4
    depends_on = ["project.cadence"]
    source_type = "llm"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        # Structural priors before LLM
        non_cm = context.non_cm_questions
        cm_sections = context.cm_section_headers

        # Journey-based: multiple CM section headers = multi-stage survey
        if len(cm_sections) >= 3:
            return TagResult(
                value="Journey-based",
                source="hybrid",
                confidence=0.85,
                evidence=ev.rule(
                    "project.relationship.multi_section",
                    f"The survey is split into {len(cm_sections)} content-message "
                    "sections. Three or more named sections means the survey walks the "
                    "respondent through distinct stages rather than asking about one "
                    "moment — that is a journey design.",
                    stage=4,
                    inputs={"section_count": len(cm_sections),
                            "sections": cm_sections[:4]},
                ),
            )

        # Pulse: short + recurring
        cadence = accumulator.get_project_tag_value("survey_cadence")
        if len(non_cm) <= 8 and cadence == "Recurring":
            return TagResult(
                value="Pulse",
                source="hybrid",
                confidence=0.80,
                evidence=ev.rule(
                    "project.relationship.short_and_recurring",
                    f"Only {len(non_cm)} real questions, and the cadence tagger already "
                    "found a recurring fielding pattern. Short plus repeated is the "
                    "definition of a pulse.",
                    stage=4,
                    inputs={"question_count": len(non_cm),
                            "survey_cadence": "Recurring"},
                ),
            )

        # Transactional signals: recent event references
        title_lower = context.survey_meta.title.lower()
        q_text = " ".join(q.title.lower() for q in non_cm[:5])
        transactional_keywords = [
            "recent visit", "recent purchase", "your visit", "your experience",
            "last 30 days", "after your", "following your", "your recent",
            "account opening", "account closure", "transaction",
        ]
        trans_matches = [kw for kw in transactional_keywords if kw in f"{title_lower} {q_text}"]
        if trans_matches:
            return TagResult(
                value="Transactional",
                source="hybrid",
                confidence=0.80,
                evidence=ev.rule(
                    "project.relationship.event_anchor",
                    "The title or opening questions anchor to a specific event the "
                    'respondent just had ("your recent visit", "after your..."). '
                    "Surveys tied to one transaction are transactional, not relational.",
                    stage=4,
                    inputs={"matched_phrases": trans_matches[:3]},
                ),
            )

        # Relational signals: NPS, overall satisfaction, no event anchor
        has_nps = context.has_nps
        has_overall = any(
            "overall" in q.title.lower() and ("satisfaction" in q.title.lower() or "rate" in q.title.lower())
            for q in non_cm
        )
        if has_nps and has_overall and not trans_matches:
            return TagResult(
                value="Relational",
                source="hybrid",
                confidence=0.75,
                evidence=ev.hybrid(
                    "project.relationship.relational_signals",
                    "Two signals point the same way and a third rules out the "
                    "alternative: the survey asks NPS and an overall-satisfaction "
                    "question, and nothing anchors it to a single event. That is a "
                    "standing measure of the whole relationship.",
                    components=[
                        ev.component("NPS question", "brand-level loyalty measure"),
                        ev.component("overall satisfaction question",
                                     "asks about the relationship, not one visit"),
                        ev.component("no event anchor",
                                     "no transactional phrasing matched"),
                    ],
                    stage=4,
                    inputs={"has_nps": True, "has_overall_satisfaction": True},
                ),
            )

        # Phase 4 prior: when no deterministic signal fires, fall back to
        # the Parallel.ai agent's relationship_type — but only when High
        # confidence (per TENANT_PROFILE_PLAN.md). Medium / Low / missing
        # falls through to the LLM placeholder below.
        profile = context.tenant_profile
        if (
            profile is not None
            and profile.has_cx
            and profile.cx_confidence == "High"
            and profile.relationship_type in _AGENT_RELATIONSHIP_MAP
        ):
            mapped = _AGENT_RELATIONSHIP_MAP[profile.relationship_type]
            return TagResult(
                value=mapped,
                source="hybrid",
                confidence=0.70,
                evidence=ev.profile(
                    "project.relationship.tenant_profile_high_conf",
                    f"No signal in the survey itself decided this, so the tenant "
                    f'profile is used: the CX agent reports a "'
                    f'{profile.relationship_type}" commercial relationship, which maps '
                    f"to {mapped}. Only High-confidence agent output is trusted this "
                    "far — Medium and Low fall through to the LLM instead.",
                    field="cx.relationship_type",
                    stage=4,
                    inputs={"agent_value": profile.relationship_type,
                            "mapped_to": mapped,
                            "agent_confidence": "High"},
                ),
            )

        # Placeholder for LLM refinement — return best guess
        return TagResult(
            value="Relational",
            source="llm",
            confidence=0.50,
            evidence=ev.fallback(
                "project.relationship.deferred_to_llm",
                "Nothing fired: fewer than three sections, not short-and-recurring, no "
                "event anchor, no NPS-plus-overall pair, and no High-confidence tenant "
                "profile. Relational is a placeholder at 0.50 for LLM Call 1 to "
                "overwrite — seeing it in the output means that call did not answer.",
                stage=4,
                inputs={"section_count": len(cm_sections),
                        "question_count": len(non_cm)},
            ),
            status="assigned",
        )


def create_tagger() -> RelationshipTagger:
    return RelationshipTagger()
