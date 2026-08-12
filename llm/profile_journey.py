"""Build + score the profile-derived journey.

Two jobs, both cheap enough to run per tenant per tagging run:

  `build_profile_journey`  reads `tenant_profile/` into a `ProfileJourney`.
      No LLM call, no disk write, no share round trip beyond the profile read
      the pipeline already does.

  `build_journey_index` / `score_questions`  embed the leaves once and rank
      them against per-question signatures. ~13 leaves is a few hundred ms on
      CPU, so the index lives in memory for the tenant's run rather than being
      persisted as an `.npz` the way the canon's was.

Scoring is deliberately self-contained rather than routed through
`llm.embeddings.score_signatures`, which is typed to `CanonEmbeddingIndex` and
belongs to the parked canon path. The shared piece — the model singleton — is
still `EmbeddingModel`, so both paths load the weights once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from models.journey import (
    JourneyLeaf,
    JourneyType,
    ProfileJourney,
    compute_source_hash,
    normalize_name,
    slugify,
)
from models.tenant_profile import TenantProfile

logger = logging.getLogger(__name__)

CX_SOURCE_FIELD = "cx.journeys[].stages[]"
EX_SOURCE_FIELD = "ex.lifecycle_analysis.stages[]"

_EX_JOURNEY_NAME = "Employee Lifecycle"


# ---------- Build from profile ----------


def build_profile_journey(
    tenant_id: int,
    profile: TenantProfile | None,
    journey_type: JourneyType,
) -> ProfileJourney | None:
    """Read one journey type out of the tenant profile.

    Returns None when the profile is absent or carries no usable stages — the
    caller then leaves the journey dimensions unassigned rather than falling
    back to a generic template. A fabricated stage that reads as grounded is
    worse for a dashboard than an explicit gap.
    """
    if profile is None:
        return None

    leaves = (
        _cx_leaves(profile) if journey_type == "CX" else _ex_leaves(profile)
    )
    if not leaves:
        return None

    journey = ProfileJourney(
        tenant_id=tenant_id,
        journey_type=journey_type,
        journey_name=_journey_name(profile, journey_type),
        leaves=leaves,
        source_field=CX_SOURCE_FIELD if journey_type == "CX" else EX_SOURCE_FIELD,
        source_hash=compute_source_hash(leaves),
    )
    logger.info(
        "profile_journey_built",
        extra={
            "tenant_id": tenant_id,
            "journey_type": journey_type,
            "leaves": len(leaves),
            "stages": len(journey.stage_values),
            "has_sub_stages": journey.has_sub_stages,
        },
    )
    return journey


def _cx_leaves(profile: TenantProfile) -> list[JourneyLeaf]:
    """CX is two-level: journey -> stage. The journey name becomes
    `journey_stage`, the stage name becomes `sub_stage_name`.

    A journey carrying no `stages[]` still yields one leaf so its questions can
    be placed at the parent level; that leaf simply has no sub-stage.
    """
    if not profile.has_cx:
        return []

    out: list[JourneyLeaf] = []
    seen: set[str] = set()
    for journey in profile.cx_journeys:
        if not isinstance(journey, dict):
            continue
        journey_name = str(journey.get("journey_name") or "").strip()
        if not journey_name:
            continue
        stages = journey.get("stages")
        stages = [s for s in stages if isinstance(s, dict)] if isinstance(stages, list) else []

        if not stages:
            _append(out, seen, JourneyLeaf(
                leaf_id=slugify(journey_name) or "stage",
                stage_value=journey_name,
                sub_stage_value=None,
                description=str(journey.get("description") or "").strip(),
            ))
            continue

        for stage in stages:
            stage_name = str(stage.get("stage_name") or stage.get("name") or "").strip()
            if not stage_name:
                continue
            _append(out, seen, JourneyLeaf(
                leaf_id=f"{slugify(journey_name)}--{slugify(stage_name)}",
                stage_value=journey_name,
                sub_stage_value=stage_name,
                description=str(stage.get("description") or "").strip(),
                goal=str(stage.get("customer_goal") or "").strip(),
            ))
    return out


def _ex_leaves(profile: TenantProfile) -> list[JourneyLeaf]:
    """EX is one-level: the lifecycle stage becomes `journey_stage` and there is
    no sub-stage. The agent schema has no second level to read, so we assign
    none — the alternative is letting the model invent one."""
    if not profile.has_ex:
        return []

    out: list[JourneyLeaf] = []
    seen: set[str] = set()
    for stage in profile.ex_lifecycle_stages:
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("stage_name") or stage.get("name") or "").strip()
        if not stage_name:
            continue
        _append(out, seen, JourneyLeaf(
            leaf_id=slugify(stage_name) or "stage",
            stage_value=stage_name,
            sub_stage_value=None,
            description=str(stage.get("description") or "").strip(),
            goal=str(stage.get("employee_goal") or stage.get("customer_goal") or "").strip(),
        ))
    return out


def _append(out: list[JourneyLeaf], seen: set[str], leaf: JourneyLeaf) -> None:
    """De-dup on the (stage, sub_stage) pair, not on the leaf name alone.

    Deliberately narrower than the canon's dedup, which collapsed same-named
    stages across different journeys and so lost the parent level. Two journeys
    may legitimately both contain a "Feedback" stage; those are different
    moments and must stay separate.
    """
    key = f"{normalize_name(leaf.stage_value)}|{normalize_name(leaf.sub_stage_value or '')}"
    if key in seen:
        return
    seen.add(key)
    out.append(leaf)


def _journey_name(profile: TenantProfile, journey_type: JourneyType) -> str:
    if journey_type == "EX":
        return _EX_JOURNEY_NAME
    industry = (profile.industry_taxonomy_vertical or profile.industry_vertical or "").strip()
    return f"{industry} Customer Journey".strip() if industry else "Customer Journey"


# ---------- Embedding index ----------


@dataclass
class JourneyIndex:
    """Leaf vectors kept beside the journey that produced them."""

    journey: ProfileJourney
    vectors: np.ndarray  # (N, dim), L2-normalized
    model_name: str = ""
    leaf_ids: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.vectors.size == 0 or not self.journey.leaves


def build_journey_index(journey: ProfileJourney, embedder) -> JourneyIndex:
    """Embed every leaf. In-memory only — nothing is written to the share."""
    if not journey.leaves:
        return JourneyIndex(journey=journey, vectors=np.zeros((0, 0), dtype=np.float32),
                            model_name=getattr(embedder, "model_name", ""))
    vectors = embedder.encode([leaf.embed_text for leaf in journey.leaves])
    return JourneyIndex(
        journey=journey,
        vectors=vectors,
        model_name=getattr(embedder, "model_name", ""),
        leaf_ids=[leaf.leaf_id for leaf in journey.leaves],
    )


def score_questions(
    signatures: Sequence[str],
    index: JourneyIndex,
    embedder,
    top_k: int = 4,
) -> list[list[tuple[JourneyLeaf, float]]]:
    """Rank leaves for each signature. One `encode()` call for the whole batch.

    Returns one ranked list per input signature, aligned by position. Vectors
    are L2-normalized at encode time, so cosine is a plain dot product.
    """
    if not signatures or index.is_empty:
        return [[] for _ in signatures]

    query = embedder.encode(list(signatures))
    if query.size == 0:
        return [[] for _ in signatures]

    scores = query @ index.vectors.T  # (Q, N)
    k = min(top_k, len(index.journey.leaves))
    out: list[list[tuple[JourneyLeaf, float]]] = []
    for row in scores:
        top_idx = np.argsort(row)[::-1][:k]
        out.append([(index.journey.leaves[i], float(row[i])) for i in top_idx])
    return out
