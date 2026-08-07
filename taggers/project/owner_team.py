"""Owner/team tagger: emits the user-applied placeholder for `owner_team`.

The auto-suggestion used to come from the account owner's name in
{TenantID}_CorporateData.json. That file is retired and its replacement
(TenantProfile, from Parallel.ai research) carries no contact identity, so
there is no owner to suggest — the dimension is emitted unvalued for the user
to fill in.
"""

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class OwnerTeamTagger(ProjectTagger):
    name = "project.owner_team"
    tag_dimension = "owner_team"
    stage = 1
    source_type = "deterministic"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        return TagResult(
            value=None,
            source="deterministic",
            confidence=1.0,
            evidence=ev.rule(
                "project.owner_team.no_identity_source",
                "The owning team used to be suggested from the account owner in "
                "{TenantID}_CorporateData.json. That file is retired and its "
                "replacement (the Parallel.ai tenant profile) carries no contact "
                "identity, so there is nothing left to infer an owner from. Emitted "
                "unvalued for a human to fill in.",
                stage=1,
                inputs={"apply_method": "User-applied"},
            ),
            apply_method="User-applied",
        )


def create_tagger() -> OwnerTeamTagger:
    return OwnerTeamTagger()
