"""AI-assist flag tagger: heuristic detection of AI-generated surveys."""

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class AIAssistTagger(ProjectTagger):
    name = "project.ai_assist"
    tag_dimension = "ai_assist_flag"
    stage = 2
    source_type = "heuristic"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        # No reliable metadata exists in current data to detect AI creation.
        # Default to Manual with low confidence.
        # Future: platform should add explicit AI-generation metadata.
        return TagResult(
            value="Manual",
            source="heuristic",
            confidence=0.50,
            evidence=ev.fallback(
                "project.ai_assist.no_platform_metadata",
                "survey_structure.json carries no AI-generation marker, and nothing in "
                "the survey body reliably distinguishes an AI-drafted survey from a "
                "hand-written one. Every survey therefore reads as Manual at 0.50 — "
                "treat this as 'not established' rather than as a finding.",
                stage=2,
            ),
        )


def create_tagger() -> AIAssistTagger:
    return AIAssistTagger()
