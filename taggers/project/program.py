"""Program/initiative tagger: placeholder for user-defined grouping."""

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class ProgramTagger(ProjectTagger):
    name = "project.program"
    tag_dimension = "program_initiative"
    stage = 1
    source_type = "deterministic"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        return TagResult(
            value=None,
            source="deterministic",
            confidence=1.0,
            apply_method="User-applied",
        )


def create_tagger() -> ProgramTagger:
    return ProgramTagger()
