"""Profile-derived journey model — the source for `journey_stage` / `sub_stage_name`.

This replaces the tenant-canon layer (parked, see `llm/tenant_canon.py`) as the
runtime grounding for the two journey dimensions. Stages are read straight off
`tenant_profile/`, which is where the canon's own inputs came from — one fewer
derived artifact to build, persist on the share, and keep in sync.

The important difference from the canon is **shape**. The canon flattened the
agent's two levels into one list, so `journey_stage` held a leaf name and
`sub_stage_name` had no grounded source at all (it defaulted to
`f"Other {stage}"` and the model filled it with metric names). The profile is
already two-level for CX, so we keep it:

    CX   agent_output.journeys[]            -> journey_stage      (Acquisition, Service, Renewal, ...)
           +- stages[]                      -> sub_stage_name     (OPD Registration, Billing & Discharge, ...)

    EX   lifecycle_analysis.stages[]        -> journey_stage      (Attraction, Recruiting, Onboarding, ...)
                                            -> sub_stage_name     (none — the source has no second level)

A `JourneyLeaf` is one scored unit: the pair the embedding ranks and the LLM
picks. Both tag values are resolved at build time onto the leaf, so nothing
downstream has to branch on CX-vs-EX or on how deep the source happened to be.
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

    @property
    def embed_text(self) -> str:
        """The text scored against a question signature.

        Carries both levels plus whatever prose the agent gave us. There are no
        LLM-generated synonyms here (the canon's canonicalization pass produced
        those and is parked), so the agent's `description` and goal do the
        semantic work — which is why both are included verbatim rather than
        summarized.
        """
        parts = [f"{self.label}."]
        if self.description:
            parts.append(self.description.strip())
        if self.goal:
            parts.append(f"Goal: {self.goal.strip()}.")
        return " ".join(parts)


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


def compute_source_hash(leaves: list[JourneyLeaf]) -> str:
    payload = [[leaf.leaf_id, leaf.stage_value, leaf.sub_stage_value or ""] for leaf in leaves]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
