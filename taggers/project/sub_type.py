"""survey_sub_type tagger — detailed classification within CX/EX.

Stage 4, hybrid+LLM. Uses only Stage 1-2 tags + content signals (per audit F1 fix —
must NOT depend on other Stage 4 taggers).
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class SurveySubTypeTagger(ProjectTagger):
    name = "project.sub_type"
    tag_dimension = "survey_sub_type"
    stage = 4
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["project.project_type", "project.audience", "project.cadence"]

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        project_type = accumulator.get_project_tag_value("project_type")
        audience = accumulator.get_project_tag_value("audience_type") or ""
        cadence = accumulator.get_project_tag_value("survey_cadence") or ""
        n_questions = len(context.non_cm_questions)

        title_lower = context.survey_meta.title.lower()
        desc_lower = (context.survey_meta.description or "").lower()
        combined = f"{title_lower} {desc_lower}"

        has_grid = any(q.question_type in ("GR", "GC", "RG", "GQ") or q.matrix_group_title
                       for q in context.questions)
        has_multiple_primary = sum(1 for q in context.questions if q.is_custom_metric or q.rs_type in (2, 3, 4)) >= 3

        # EX project_type
        if project_type == "EX":
            if any(kw in combined for kw in ("pulse", "weekly", "bi-weekly")):
                return TagResult(value="Pulse", source="hybrid", confidence=0.80,
                                 evidence="EX + pulse keyword")
            if any(kw in combined for kw in ("onboarding", "exit", "anniversary", "stay interview", "offboarding")):
                return TagResult(value="Lifecycle", source="hybrid", confidence=0.80,
                                 evidence="EX + lifecycle keyword")
            if any(kw in combined for kw in ("360", "multi-rater", "360-degree")):
                return TagResult(value="360 Feedback", source="hybrid", confidence=0.85,
                                 evidence="EX + 360 keyword")
            if any(kw in combined for kw in ("manager effectiveness", "manager feedback", "leadership")):
                return TagResult(value="Manager Effectiveness", source="hybrid",
                                 confidence=0.75, evidence="EX + manager keyword")
            if any(kw in combined for kw in ("culture", "belonging", "values", "dei")):
                return TagResult(value="Culture", source="hybrid", confidence=0.75,
                                 evidence="EX + culture keyword")
            if context.has_nps or any("engagement" in combined for _ in [0]) or n_questions >= 15:
                return TagResult(value="Engagement", source="hybrid", confidence=0.65,
                                 evidence=f"EX + eNPS/long ({n_questions} qs)")
            return TagResult(value="Engagement", source="hybrid", confidence=0.50,
                             evidence="EX default")

        # CX project_type
        if project_type == "CX":
            if any(kw in combined for kw in ("journey", "lifecycle", "multi-stage")):
                return TagResult(value="Journey", source="hybrid", confidence=0.80,
                                 evidence="CX + journey keyword")
            if any(kw in combined for kw in ("transaction", "post-", "after-", "delivery", "service call")):
                return TagResult(value="Transactional", source="hybrid", confidence=0.75,
                                 evidence="CX + transactional keyword")
            if any(kw in combined for kw in ("touchpoint", "website", "store", "kiosk", "call center")):
                return TagResult(value="Touchpoint", source="hybrid", confidence=0.75,
                                 evidence="CX + touchpoint keyword")
            if any(kw in combined for kw in ("product", "feature", "service quality")):
                return TagResult(value="Product / Service", source="hybrid",
                                 confidence=0.65, evidence="CX + product keyword")
            if cadence in ("Recurring", "Always-on"):
                return TagResult(value="Relationship", source="hybrid", confidence=0.60,
                                 evidence=f"CX + {cadence} cadence")
            return TagResult(value="Transactional", source="hybrid", confidence=0.55,
                             evidence="CX default")

        # Survey / Assessment
        if project_type in ("Survey", "Assessment"):
            if "market research" in combined or "voc" in combined:
                return TagResult(value="Market Research", source="hybrid",
                                 confidence=0.80, evidence="Market research keyword")
            return TagResult(value="Ad Hoc", source="hybrid", confidence=0.55,
                             evidence=f"project_type={project_type} default")

        return TagResult(value="Ad Hoc", source="hybrid", confidence=0.40,
                         evidence="No project_type match")


def create_tagger() -> SurveySubTypeTagger:
    return SurveySubTypeTagger()
