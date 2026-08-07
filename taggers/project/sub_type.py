"""survey_sub_type tagger — detailed classification within CX/EX.

Stage 4, hybrid+LLM. Uses only Stage 1-2 tags + content signals (per audit F1 fix —
must NOT depend on other Stage 4 taggers).
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


def _first_match(haystack: str, keywords: tuple[str, ...]) -> str | None:
    """The keyword that actually fired, so the evidence can name it rather than
    reprinting the whole candidate list."""
    for kw in keywords:
        if kw in haystack:
            return kw
    return None


def _keyword_evidence(rule_suffix: str, matched: str, value: str,
                      project_type: str, title: str) -> dict:
    return ev.rule(
        f"project.sub_type.{rule_suffix}",
        f'The survey is typed {project_type} and its title or description contains '
        f'"{matched}", which this tagger reads as a {value} programme.',
        stage=4,
        inputs={"project_type": project_type, "matched_keyword": matched},
        quote=title,
    )


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

        title = context.survey_meta.title

        # EX project_type
        if project_type == "EX":
            if (m := _first_match(combined, ("pulse", "weekly", "bi-weekly"))):
                return TagResult(value="Pulse", source="hybrid", confidence=0.80,
                                 evidence=_keyword_evidence("ex_pulse", m, "Pulse",
                                                            "EX", title))
            if (m := _first_match(combined, ("onboarding", "exit", "anniversary",
                                             "stay interview", "offboarding"))):
                return TagResult(value="Lifecycle", source="hybrid", confidence=0.80,
                                 evidence=_keyword_evidence("ex_lifecycle", m,
                                                            "Lifecycle", "EX", title))
            if (m := _first_match(combined, ("360", "multi-rater", "360-degree"))):
                return TagResult(value="360 Feedback", source="hybrid", confidence=0.85,
                                 evidence=_keyword_evidence("ex_360", m, "360 Feedback",
                                                            "EX", title))
            if (m := _first_match(combined, ("manager effectiveness", "manager feedback",
                                             "leadership"))):
                return TagResult(value="Manager Effectiveness", source="hybrid",
                                 confidence=0.75,
                                 evidence=_keyword_evidence("ex_manager", m,
                                                            "Manager Effectiveness",
                                                            "EX", title))
            if (m := _first_match(combined, ("culture", "belonging", "values", "dei"))):
                return TagResult(value="Culture", source="hybrid", confidence=0.75,
                                 evidence=_keyword_evidence("ex_culture", m, "Culture",
                                                            "EX", title))
            if context.has_nps or "engagement" in combined or n_questions >= 15:
                triggers = []
                if context.has_nps:
                    triggers.append(ev.component("eNPS question",
                                                 "the standard engagement headline metric"))
                if "engagement" in combined:
                    triggers.append(ev.component('"engagement" in title/description'))
                if n_questions >= 15:
                    triggers.append(ev.component(f"{n_questions} questions",
                                                 "long enough to be a full engagement "
                                                 "study rather than a pulse"))
                return TagResult(
                    value="Engagement", source="hybrid", confidence=0.65,
                    evidence=ev.hybrid(
                        "project.sub_type.ex_engagement",
                        "No pulse, lifecycle, 360, manager or culture keyword matched, "
                        "but this EX survey looks like a full engagement study for the "
                        "reason(s) listed.",
                        components=triggers,
                        stage=4,
                        inputs={"project_type": "EX", "question_count": n_questions},
                    ),
                )
            return TagResult(
                value="Engagement", source="hybrid", confidence=0.50,
                evidence=ev.fallback(
                    "project.sub_type.ex_default",
                    f"An EX survey that matched no sub-type keyword, has no eNPS, and "
                    f"is only {n_questions} question(s) long. Engagement is the EX "
                    "default rather than a finding — the 0.50 leaves it open for LLM "
                    "Call 1.",
                    stage=4,
                    inputs={"project_type": "EX", "question_count": n_questions},
                ),
            )

        # CX project_type
        if project_type == "CX":
            if (m := _first_match(combined, ("journey", "lifecycle", "multi-stage"))):
                return TagResult(value="Journey", source="hybrid", confidence=0.80,
                                 evidence=_keyword_evidence("cx_journey", m, "Journey",
                                                            "CX", title))
            if (m := _first_match(combined, ("transaction", "post-", "after-",
                                             "delivery", "service call"))):
                return TagResult(value="Transactional", source="hybrid", confidence=0.75,
                                 evidence=_keyword_evidence("cx_transactional", m,
                                                            "Transactional", "CX", title))
            if (m := _first_match(combined, ("touchpoint", "website", "store", "kiosk",
                                             "call center"))):
                return TagResult(value="Touchpoint", source="hybrid", confidence=0.75,
                                 evidence=_keyword_evidence("cx_touchpoint", m,
                                                            "Touchpoint", "CX", title))
            if (m := _first_match(combined, ("product", "feature", "service quality"))):
                return TagResult(value="Product / Service", source="hybrid",
                                 confidence=0.65,
                                 evidence=_keyword_evidence("cx_product", m,
                                                            "Product / Service",
                                                            "CX", title))
            if cadence in ("Recurring", "Always-on"):
                return TagResult(
                    value="Relationship", source="hybrid", confidence=0.60,
                    evidence=ev.rule(
                        "project.sub_type.cx_cadence",
                        f"No sub-type keyword matched, but the cadence tagger found a "
                        f"{cadence} fielding pattern. A CX survey that runs "
                        "continuously is measuring the standing relationship, not one "
                        "transaction.",
                        stage=4,
                        inputs={"project_type": "CX", "survey_cadence": cadence},
                    ),
                )
            return TagResult(
                value="Transactional", source="hybrid", confidence=0.55,
                evidence=ev.fallback(
                    "project.sub_type.cx_default",
                    f"A CX survey that matched no sub-type keyword and whose cadence "
                    f"({cadence or 'unknown'}) is not recurring or always-on. "
                    "Transactional is the CX default rather than a finding.",
                    stage=4,
                    inputs={"project_type": "CX",
                            "survey_cadence": cadence or "(unset)"},
                ),
            )

        # Survey / Assessment
        if project_type in ("Survey", "Assessment"):
            if (m := _first_match(combined, ("market research", "voc"))):
                return TagResult(value="Market Research", source="hybrid",
                                 confidence=0.80,
                                 evidence=_keyword_evidence("market_research", m,
                                                            "Market Research",
                                                            project_type, title))
            return TagResult(
                value="Ad Hoc", source="hybrid", confidence=0.55,
                evidence=ev.fallback(
                    "project.sub_type.generic_default",
                    f"project_type is {project_type}, which has no sub-type taxonomy of "
                    "its own beyond Market Research, and no market-research keyword "
                    "matched. Ad Hoc is the catch-all.",
                    stage=4,
                    inputs={"project_type": project_type},
                ),
            )

        return TagResult(
            value="Ad Hoc", source="hybrid", confidence=0.40,
            evidence=ev.fallback(
                "project.sub_type.no_project_type",
                f"project_type came back as {project_type or '(unset)'} — none of EX, "
                "CX, Survey or Assessment — so no sub-type branch applies at all. Check "
                "the project_type tag before trusting this one.",
                stage=4,
                inputs={"project_type": project_type or "(unset)"},
            ),
        )


def create_tagger() -> SurveySubTypeTagger:
    return SurveySubTypeTagger()
