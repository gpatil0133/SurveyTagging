"""Profile-derived journey model — the source for `journey_stage` / `sub_stage_name`.

This replaced the tenant-canon layer (since deleted) as the runtime grounding
for the two journey dimensions. Stages are read straight off `tenant_profile/`,
which is where the canon's own inputs came from — one fewer derived artifact to
build, persist on the share, and keep in sync.

The important difference from the canon is **shape**. The canon flattened the
agent's two levels into one list, so `journey_stage` held a leaf name and
`sub_stage_name` had no grounded source at all (it defaulted to
`f"Other {stage}"` and the model filled it with metric names). The profile is
already two-level for CX, so we keep it:

    CX   agent_output.journeys[]            -> journey_stage      (Acquisition, Service, Renewal, ...)
           +- stages[]                      -> sub_stage_name     (OPD Registration, Billing & Discharge, ...)

    EX   lifecycle_analysis.stages[]        -> journey_stage      (Attraction, Recruiting, Onboarding, ...)
                                            -> sub_stage_name     (none — the source has no second level)

A `JourneyLeaf` is one selectable unit: the pair the LLM picks from. Both tag
values are resolved at build time onto the leaf, so nothing downstream has to
branch on CX-vs-EX or on how deep the source happened to be.

V9: the whole leaf set is inlined into the question prompt and the model selects
from it directly. There is no retrieval step — the embedding index that used to
cut this list to a top-4 per question was removed, because at real journey sizes
it cost more prompt tokens than it saved (the cut list was repeated per question;
the full list is sent once) and it could exclude the correct leaf outright.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

JourneyType = Literal["CX", "EX"]


def slugify(text: str) -> str:
    """Lowercase alnum + single dashes. Shared with leaf-id construction."""
    out: list[str] = []
    prev_dash = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash and out:
            out.append("-")
            prev_dash = True
    return "".join(out).rstrip("-")


def normalize_name(name: str) -> str:
    """Case/punctuation-insensitive key so "OPD Registration", "OPD-Registration"
    and "opd_registration" collapse to one leaf."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()


class JourneyLeaf(BaseModel):
    """One scoreable moment in the tenant's journey.

    `stage_value` and `sub_stage_value` are the literal tag values this leaf
    produces — resolved here rather than downstream so the CX (two-level) and
    EX (one-level) shapes present one interface. `sub_stage_value` is None when
    the source has no second level; that is a real "no value", not a gap to be
    filled by the model.
    """

    leaf_id: str
    stage_value: str
    sub_stage_value: str | None = None
    description: str = ""
    goal: str = ""

    @property
    def label(self) -> str:
        """Human-readable path, for prompts and log lines."""
        if self.sub_stage_value:
            return f"{self.stage_value} > {self.sub_stage_value}"
        return self.stage_value

    def as_prompt_entry(self) -> dict:
        """The leaf as the model sees it in the inlined journey catalog.

        `description` and `goal` carry the agent's own prose verbatim: they are
        what lets the model tell two similarly-named moments apart, and there
        are no LLM-generated synonyms to lean on (the canon's canonicalization
        pass produced those and is parked). Empty strings are dropped rather
        than sent as `""` — an absent key reads as "not stated", which is true,
        and it keeps the catalog small.
        """
        entry = {"leaf_id": self.leaf_id, "stage_name": self.stage_value}
        if self.sub_stage_value:
            entry["sub_stage_name"] = self.sub_stage_value
        if self.description:
            entry["description"] = self.description.strip()
        if self.goal:
            entry["goal"] = self.goal.strip()
        return entry


class ProfileJourney(BaseModel):
    """A tenant's journey for one journey type, read from `tenant_profile/`."""

    tenant_id: int
    journey_type: JourneyType
    journey_name: str
    leaves: list[JourneyLeaf] = Field(default_factory=list)
    # Dotted path inside the profile envelope the leaves came from — carried
    # into tag evidence so an operator can go verify the value at its source.
    source_field: str = ""
    # Fingerprint of the leaves; a cache-key input for the question LLM call.
    source_hash: str = ""

    @property
    def has_sub_stages(self) -> bool:
        return any(leaf.sub_stage_value for leaf in self.leaves)

    @property
    def stage_values(self) -> list[str]:
        """Distinct `journey_stage` values, in source order."""
        seen: dict[str, None] = {}
        for leaf in self.leaves:
            seen[leaf.stage_value] = None
        return list(seen)

    def leaf(self, leaf_id: str) -> JourneyLeaf | None:
        for candidate in self.leaves:
            if candidate.leaf_id == leaf_id:
                return candidate
        return None

    def catalog(self) -> list[dict]:
        """Every leaf, in source order, as the prompt's selection list.

        Source order rather than any ranking: the model is shown the tenant's
        journey as the tenant wrote it, and a synthetic order would read as a
        recommendation the pipeline is in no position to make.
        """
        return [leaf.as_prompt_entry() for leaf in self.leaves]

    def match_by_name(self, stage: str | None, sub_stage: str | None) -> JourneyLeaf | None:
        """Find a leaf by its literal names — the second chance for a model that
        wrote the moment out instead of copying its `leaf_id`.

        Matches the (stage, sub_stage) pair when both were given, else whichever
        single name was. Case-insensitive and punctuation-insensitive via
        `normalize_name`, but never fuzzy: a near miss returns None and is
        reported as unresolved rather than filed under a guess.
        """
        stage_key = normalize_name(stage) if isinstance(stage, str) else ""
        sub_key = normalize_name(sub_stage) if isinstance(sub_stage, str) else ""
        if not stage_key and not sub_key:
            return None

        for leaf in self.leaves:
            leaf_stage = normalize_name(leaf.stage_value)
            leaf_sub = normalize_name(leaf.sub_stage_value or "")
            if stage_key and sub_key:
                if leaf_stage == stage_key and leaf_sub == sub_key:
                    return leaf
            elif stage_key:
                # Only safe when it identifies ONE leaf: a two-level journey
                # legitimately has several moments under the same stage, and
                # picking the first would silently invent a sub-stage.
                hits = [c for c in self.leaves
                        if normalize_name(c.stage_value) == stage_key]
                return hits[0] if len(hits) == 1 else None
            elif sub_key and leaf_sub == sub_key:
                return leaf
        return None


def compute_source_hash(leaves: list[JourneyLeaf]) -> str:
    payload = [[leaf.leaf_id, leaf.stage_value, leaf.sub_stage_value or ""] for leaf in leaves]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
