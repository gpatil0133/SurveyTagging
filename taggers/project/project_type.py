"""Project type tagger: maps surveyType to taxonomy project_type.

V6: renamed from `category` (file was `category.py`, class was `CategoryTagger`,
dimension was `category`). Allowed values are unchanged.
"""

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

_PROJECT_TYPE_MAP = {
    "Survey": "Survey",
    "CX": "CX",
    "EX": "EX",
    "Assessment": "Assessment",
    "Poll": "Survey",
}


class ProjectTypeTagger(ProjectTagger):
    name = "project.project_type"
    tag_dimension = "project_type"
    stage = 1
    source_type = "deterministic"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        survey_type = context.survey_meta.survey_type
        mapped = _PROJECT_TYPE_MAP.get(survey_type)
        value = mapped or "Survey"

        if mapped is not None:
            evidence = ev.rule(
                "project.project_type.platform_type_map",
                f'The platform records this survey\'s type as "{survey_type}", which '
                f"maps directly onto the {value} taxonomy value.",
                stage=1,
                inputs={"survey_type": survey_type},
            )
        else:
            # Note the confidence stays 1.0 here even though nothing was
            # recognized — the typed `fallback` marker is what lets an auditor
            # find these rather than reading them as a positive classification.
            evidence = ev.fallback(
                "project.project_type.unknown_platform_type",
                f'The platform reported survey type "{survey_type or "(empty)"}", which '
                "is not one of Survey / CX / EX / Assessment / Poll. Defaulted to "
                "Survey — the value is a default, not a classification.",
                stage=1,
                inputs={"survey_type": survey_type or "(empty)",
                        "known_types": sorted(_PROJECT_TYPE_MAP)},
            )

        return TagResult(value=value, source="deterministic", confidence=1.0,
                         evidence=evidence)


def create_tagger() -> ProjectTypeTagger:
    return ProjectTypeTagger()
