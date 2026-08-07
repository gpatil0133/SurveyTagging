"""AI-assist flag tagger: heuristic detection of AI-generated surveys."""

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
            evidence="No AI-generation metadata available; defaulting to Manual",
        )


def create_tagger() -> AIAssistTagger:
    return AIAssistTagger()
