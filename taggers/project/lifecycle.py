"""Lifecycle status tagger: determines Edit/Active/Expired from dates and response data."""

from datetime import datetime

from models import evidence as ev
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
                return TagResult(
                    value="Active", source="deterministic", confidence=0.90,
                    evidence=ev.rule(
                        "project.lifecycle.no_dates_with_responses",
                        "Neither a start nor an end date is set, but responses have "
                        "already come in — a survey that is collecting is live "
                        "regardless of what its date fields say.",
                        stage=1,
                        inputs={"start_date": "(unset)", "end_date": "(unset)",
                                "has_responses": True},
                    ),
                )
            return TagResult(
                value="Edit", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "project.lifecycle.no_dates_no_responses",
                    "No start date, no end date and no responses — nothing indicates "
                    "this survey has ever been fielded, so it reads as still in Edit.",
                    stage=1,
                    inputs={"start_date": "(unset)", "end_date": "(unset)",
                            "has_responses": False},
                ),
            )

        if start and start > now:
            return TagResult(
                value="Edit", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "project.lifecycle.start_in_future",
                    f"The start date ({start.date()}) has not arrived yet, so the "
                    "survey cannot be collecting.",
                    stage=1,
                    inputs={"start_date": str(start.date()),
                            "evaluated_on": str(now.date())},
                ),
            )

        if end and end < now:
            return TagResult(
                value="Expired", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "project.lifecycle.end_passed",
                    f"The end date ({end.date()}) is in the past, so collection has "
                    "closed.",
                    stage=1,
                    inputs={"end_date": str(end.date()),
                            "evaluated_on": str(now.date())},
                ),
            )

        return TagResult(
            value="Active", source="deterministic", confidence=1.0,
            evidence=ev.rule(
                "project.lifecycle.window_open",
                "Today falls inside the survey's collection window — the start date "
                "has passed (or none is set) and the end date has not.",
                stage=1,
                inputs={"start_date": str(start.date()) if start else "(unset)",
                        "end_date": str(end.date()) if end else "(unset)",
                        "evaluated_on": str(now.date())},
            ),
        )


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
