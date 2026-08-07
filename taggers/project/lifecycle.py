"""Lifecycle status tagger: determines Edit/Active/Expired from dates and response data."""

from datetime import datetime

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class LifecycleTagger(ProjectTagger):
    name = "project.lifecycle"
    tag_dimension = "lifecycle_status"
    stage = 1
    source_type = "deterministic"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        now = datetime.now()
        start = _parse_date(context.survey_meta.start_date)
        end = _parse_date(context.survey_meta.end_date)
        has_responses = context.has_responses

        if start is None and end is None:
            if has_responses:
                return TagResult(value="Active", source="deterministic", confidence=0.90,
                                 evidence="No dates set but responses exist")
            else:
                return TagResult(value="Edit", source="deterministic", confidence=0.85,
                                 evidence="No dates set and no responses")

        if start and start > now:
            return TagResult(value="Edit", source="deterministic", confidence=1.0,
                             evidence=f"Start date {start.date()} is in the future")

        if end and end < now:
            return TagResult(value="Expired", source="deterministic", confidence=1.0,
                             evidence=f"End date {end.date()} has passed")

        return TagResult(value="Active", source="deterministic", confidence=1.0)


def _parse_date(val: str | None) -> datetime | None:
    if not val:
        return None
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def create_tagger() -> LifecycleTagger:
    return LifecycleTagger()
