"""Project type tagger: maps surveyType to taxonomy project_type.

V6: renamed from `category` (file was `category.py`, class was `CategoryTagger`,
dimension was `category`). Allowed values are unchanged.
"""

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
        value = _PROJECT_TYPE_MAP.get(survey_type, "Survey")
        return TagResult(value=value, source="deterministic", confidence=1.0)


def create_tagger() -> ProjectTypeTagger:
    return ProjectTypeTagger()
