"""dashboard_placement tagger — which dashboards this question appears on.

Stage 5, multi-label, hybrid+LLM. Uses only Stage 1-4 tags
(per audit F1 — no same-stage dependencies).

Constrains values to project-level `dashboard_routing` (which is set at Stage 4).
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class DashboardPlacementTagger(QuestionTagger):
    name = "question.dashboard_placement"
    tag_dimension = "dashboard_placement"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent", "project.dashboard_routing"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=[], source="deterministic", status="skipped",
                             evidence=ev.content_message("dashboard_placement", stage=5))

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")
        project_type = accumulator.get_project_tag_value("project_type")
        project_dashboards = accumulator.get_project_tag_value("dashboard_routing") or []
        if not isinstance(project_dashboards, list):
            project_dashboards = [project_dashboards]

        placements: list[str] = []
        # Multi-label: track why each dashboard was chosen, and also why a rule
        # wanted one it could not have.
        why: list[dict] = []

        def maybe_add(name: str, reason: str) -> None:
            if name not in project_dashboards:
                # A rule fired but the survey does not route to that dashboard.
                # Worth recording — it is the usual reason a question lands
                # somewhere unexpected.
                why.append(ev.component(
                    name, f"{reason}, but the survey does not route to this dashboard"))
                return
            if name not in placements:
                placements.append(name)
                why.append(ev.component(name, reason))

        # Standard Metric (NPS/CSAT/CES/eNPS) + Primary Metric → Executive Dashboard
        if metric_type == "Standard Metric" and role == "Primary Metric":
            maybe_add("Executive Dashboard",
                      "a standard metric in the primary role — the headline number")

        # Open-ended text → VoC/VoE dashboards
        if metric_type == "Open-ended":
            if project_type == "CX":
                maybe_add("Voice of Customer Dashboard",
                          "open-ended text on a CX survey")
            elif project_type == "EX":
                maybe_add("Voice of Employee Dashboard",
                          "open-ended text on an EX survey")

        # Drivers → project_type-specific dashboard
        if role == "Driver / Attribute":
            if project_type == "CX":
                maybe_add("Customer Experience Dashboard",
                          "a driver question on a CX survey")
            elif project_type == "EX":
                maybe_add("Employee Engagement Dashboard",
                          "a driver question on an EX survey")

        # If nothing matched, default to first project-level dashboard so every
        # question has at least one placement (LLM refines).
        signals = {"metric_type": metric_type or "(unset)",
                   "role_intent": role or "(unset)",
                   "project_type": project_type or "(unset)",
                   "project_dashboards": project_dashboards}

        if not placements:
            if project_dashboards:
                placements = [project_dashboards[0]]
                return TagResult(
                    value=placements, source="hybrid", confidence=0.60,
                    evidence=ev.fallback(
                        "question.dashboard_placement.first_routed_default",
                        f"No placement rule fired for this question, so it defaults to "
                        f"the survey's first routed dashboard "
                        f"({project_dashboards[0]}) rather than appearing nowhere. LLM "
                        "Call 2 usually replaces this.",
                        stage=5,
                        inputs=signals,
                    ),
                )
            return TagResult(
                value=placements, source="hybrid", confidence=0.60,
                evidence=ev.fallback(
                    "question.dashboard_placement.no_routed_dashboards",
                    "The survey's project-level dashboard_routing is empty, so there "
                    "is nowhere to place this question. Fix dashboard_routing first — "
                    "this tag can only choose from that list.",
                    stage=5,
                    inputs=signals,
                ),
            )

        return TagResult(
            value=placements, source="hybrid", confidence=0.60,
            evidence=ev.hybrid(
                "question.dashboard_placement.rules",
                f"{len(placements)} placement(s) chosen from the survey's routed "
                "dashboards. Each entry below names a dashboard and the rule behind "
                "it; entries noting the survey does not route there are rules that "
                "fired but could not be honoured.",
                components=why,
                stage=5,
                inputs=signals,
            ),
        )


def create_tagger() -> DashboardPlacementTagger:
    return DashboardPlacementTagger()
