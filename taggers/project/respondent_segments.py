"""respondent_segments tagger — the attributes this survey's results can be cut by.

Stage 2, deterministic. No tag dependencies.

The gap this closes: the tenant's directories carry Department, Location, City,
Gender, Membership/Employment Status and more, and `directory_linking.parquet`
joins each response to a directory record — so "CSAT by branch", "eNPS by
department" and "satisfaction by membership status" are all answerable from data
already on the share. Nothing said so. `loaders/directory.py` read directory column
NAMES and used them for one thing (guessing industry_vertical), and
`segment_dimensions` is question-level, so it can only ever say which QUESTION
segments others — never which respondent attributes exist.

Shape is `[{category, values[]}]`, matching what the widget layer already speaks
(`TagDetails{categoryName, tagName}`), so a consumer can render a segment picker
straight from this without a second lookup.

Emitted only when the survey is actually linked to a directory. An unlinked survey
gets `status="skipped"` with evidence naming the missing file, because attributes
that cannot reach a response are not segments — offering them would promise a
breakdown the dashboard could not build.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

# Attributes worth naming in the evidence as the ones a reader will look for
# first. Ordering only — every candidate is emitted regardless.
_HEADLINE = ("department", "location", "city", "job title", "gender",
             "membership status", "employment status")


def _sort_key(category: str) -> tuple[int, str]:
    low = category.lower()
    for index, name in enumerate(_HEADLINE):
        if name in low:
            return (index, low)
    return (len(_HEADLINE), low)


class RespondentSegmentsTagger(ProjectTagger):
    name = "project.respondent_segments"
    tag_dimension = "respondent_segments"
    stage = 2
    source_type = "deterministic"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        segments = context.respondent_segments

        if not context.has_linking:
            return TagResult(
                value=[], source="deterministic", status="skipped", confidence=1.0,
                evidence=ev.rule(
                    "project.respondent_segments.no_directory_link",
                    "This survey has no directory_linking.parquet, so no response can "
                    "be traced to a directory record. The tenant's directory "
                    "attributes describe people whose answers cannot be identified "
                    "here, which makes them unusable as segments for THIS survey.",
                    stage=2,
                    inputs={"has_linking": False,
                            "tenant_directories": context.directory_signals.directory_ids},
                ),
            )

        if not segments:
            reason = (
                "linked to directory %s, but none of its columns qualifies as a "
                "segment: every column is either an identifier, a name or contact "
                "detail (never emitted), or has too many distinct values to group by."
                % ", ".join(context.linked_directory_ids or ["(unknown)"])
            )
            return TagResult(
                value=[], source="deterministic", status="skipped", confidence=0.90,
                evidence=ev.rule(
                    "project.respondent_segments.no_usable_attributes",
                    "This survey is " + reason,
                    stage=2,
                    inputs={"linked_directory_ids": context.linked_directory_ids,
                            "tenant_directories": context.directory_signals.directory_ids},
                ),
            )

        value = [
            {"category": category, "values": segments[category]}
            for category in sorted(segments, key=_sort_key)
        ]
        named = ", ".join(f"{s['category']} ({len(s['values'])})" for s in value[:6])

        return TagResult(
            value=value,
            source="deterministic",
            confidence=1.0,
            evidence=ev.rule(
                "project.respondent_segments.from_linked_directory",
                f"{len(value)} respondent attribute(s) usable as segments, read from "
                f"directory {', '.join(context.linked_directory_ids)} which this "
                f"survey's responses are joined to: {named}"
                + (", ..." if len(value) > 6 else ".")
                + " Each carries its distinct values, so a report can offer the "
                  "breakdown directly. Identifiers, names and contact details are "
                  "excluded by rule, and only attributes with 2-15 distinct values "
                  "qualify — past that the groups get too thin to read.",
                stage=2,
                inputs={"linked_directory_ids": context.linked_directory_ids,
                        "categories": [s["category"] for s in value],
                        "value_counts": {s["category"]: len(s["values"]) for s in value}},
            ),
        )


def create_tagger() -> RespondentSegmentsTagger:
    return RespondentSegmentsTagger()
