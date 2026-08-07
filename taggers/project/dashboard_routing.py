"""dashboard_routing tagger — which dashboards this survey should appear on.

Stage 4, multi-label, hybrid+LLM. Uses only Stage 1-2 tags + context signals
(per audit F1 fix — must NOT depend on other Stage 4 taggers).
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class DashboardRoutingTagger(ProjectTagger):
    name = "project.dashboard_routing"
    tag_dimension = "dashboard_routing"
    stage = 4
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["project.project_type", "project.audience"]

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        project_type = accumulator.get_project_tag_value("project_type")
        audience = accumulator.get_project_tag_value("audience_type") or ""

        dashboards: list[str] = []
        # Why each dashboard got on the list — this is a multi-label tag, so a
        # single sentence about the whole list explains none of the entries.
        why: list[dict] = []

        # Executive Dashboard if standard headline metric present
        if context.has_nps or context.has_csat:
            dashboards.append("Executive Dashboard")
            metric = "NPS" if context.has_nps else "CSAT"
            why.append(ev.component(
                "Executive Dashboard",
                f"survey carries a headline {metric} question",
            ))

        # project_type-based dashboards
        if project_type == "EX":
            dashboards.append("Employee Engagement Dashboard")
            why.append(ev.component("Employee Engagement Dashboard",
                                    "project_type is EX"))
            if "Employees" in audience:
                dashboards.append("HR Dashboard")
                why.append(ev.component("HR Dashboard",
                                        f"audience_type is {audience!r}"))
        elif project_type == "CX":
            dashboards.append("Customer Experience Dashboard")
            why.append(ev.component("Customer Experience Dashboard",
                                    "project_type is CX"))

        # Voice-of-* dashboard if there are open-ended questions
        has_text = any(q.question_type == "T" and not q.is_content_message
                       for q in context.questions)
        if has_text:
            if project_type == "CX":
                dashboards.append("Voice of Customer Dashboard")
                why.append(ev.component("Voice of Customer Dashboard",
                                        "CX survey with open-ended questions"))
            elif project_type == "EX":
                dashboards.append("Voice of Employee Dashboard")
                why.append(ev.component("Voice of Employee Dashboard",
                                        "EX survey with open-ended questions"))

        # Dedupe while preserving insertion order
        seen = set()
        deduped = []
        for d in dashboards:
            if d not in seen:
                deduped.append(d)
                seen.add(d)

        if not deduped:
            # Safety floor: always emit at least one canonical dashboard
            if project_type == "EX":
                deduped = ["Employee Engagement Dashboard"]
            else:
                deduped = ["Customer Experience Dashboard"]
            return TagResult(
                value=deduped, source="hybrid", confidence=0.70,
                evidence=ev.fallback(
                    "project.dashboard_routing.safety_floor",
                    f"No routing rule fired — project_type is "
                    f"{project_type or '(unset)'}, there is no NPS or CSAT question, "
                    f"and no open-ended question. Emitted {deduped[0]} as the floor so "
                    "the survey lands somewhere rather than nowhere.",
                    stage=4,
                    inputs={"project_type": project_type or "(unset)",
                            "has_nps": context.has_nps,
                            "has_csat": context.has_csat,
                            "has_text_questions": has_text},
                ),
            )

        return TagResult(
            value=deduped, source="hybrid", confidence=0.70,
            evidence=ev.hybrid(
                "project.dashboard_routing.rules",
                f"{len(deduped)} dashboard(s) selected by routing rules over "
                f"project_type, audience and the survey's question mix. Each entry "
                "below names the dashboard and the rule that placed it. The LLM may "
                "add to this list afterwards.",
                components=why,
                stage=4,
                inputs={"project_type": project_type or "(unset)",
                        "audience_type": audience or "(unset)",
                        "has_nps": context.has_nps,
                        "has_csat": context.has_csat,
                        "has_text_questions": has_text},
            ),
        )


def create_tagger() -> DashboardRoutingTagger:
    return DashboardRoutingTagger()
