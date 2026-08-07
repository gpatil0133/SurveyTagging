"""Relationship type tagger: LLM-based with structural priors."""

from __future__ import annotations

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
                evidence=f"Multiple survey sections: {cm_sections[:4]}",
            )

        # Pulse: short + recurring
        cadence = accumulator.get_project_tag_value("survey_cadence")
        if len(non_cm) <= 8 and cadence == "Recurring":
            return TagResult(
                value="Pulse",
                source="hybrid",
                confidence=0.80,
                evidence=f"Short survey ({len(non_cm)} questions) with recurring cadence",
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
                evidence=f"Transactional keywords: {trans_matches[:3]}",
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
                evidence="NPS + overall satisfaction without event anchor",
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
                evidence=(
                    f"Parallel.ai cx.relationship_type={profile.relationship_type!r} "
                    f"(High confidence) → {mapped}"
                ),
            )

        # Placeholder for LLM refinement — return best guess
        return TagResult(
            value="Relational",
            source="llm",
            confidence=0.50,
            evidence="Requires LLM classification",
            status="assigned",
        )


def create_tagger() -> RelationshipTagger:
    return RelationshipTagger()
