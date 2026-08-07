"""Program/initiative tagger: placeholder for user-defined grouping."""

from models import evidence as ev
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
            evidence=ev.rule(
                "project.program.user_applied_placeholder",
                "program_initiative is a free-text grouping only the customer can "
                "name — no signal in the survey, corporate data or tenant profile "
                "identifies which internal program a survey belongs to. The dimension "
                "is emitted unvalued for a human to fill in.",
                stage=1,
                inputs={"apply_method": "User-applied"},
            ),
            apply_method="User-applied",
        )


def create_tagger() -> ProgramTagger:
    return ProgramTagger()
