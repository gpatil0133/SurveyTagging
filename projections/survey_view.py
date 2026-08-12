"""Survey view projection — unified per-survey response shaped for API consumers.

Input: a tagged_output.json dict (or path).
Output: a dict with:
  - `project_tags`: passthrough of the 12 project-level dimensions (with
    `category` already renamed to `project_type` in V6).
  - `questions`: list of per-question tag entries; each carries `journey_stage`
    and `sub_stage_name` tags inline alongside the other 21 question dimensions.
  - `survey_journey`: a small derived rollup — which journey stages and
    sub-stages the survey touches, derived from the question tags. Present
    only when `project_tags.project_type.value` ∈ {"CX", "EX"}.

Pure function. No filesystem I/O, no LLM, no journey-artifact reads. This is
the join-free, fast endpoint payload. The richer cross-survey journey context
(stage definitions, sibling questions) stays under `/api/projections/custom-journey/...`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

from projections._tag_utils import get_tag_value

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "5.0"

_JOURNEY_TYPES = {"CX", "EX"}

# Dimensions that carry `coverage_metadata` (the per-question canon candidates
# scored by embeddings + LLM confidence). We strip these by default for payload
# size; surface them when `include_journey_candidates=True`.
_COVERAGE_DIMS = ("journey_stage", "sub_stage_name")


def build_survey_view(
    tagged: dict,
    *,
    include_journey_candidates: bool = False,
) -> dict:
    """Build the survey-view payload from a tagged_output.json dict.

    Args:
        tagged: raw `tagged_output.json` dict.
        include_journey_candidates: when True, the `journey_stage` and
            `sub_stage_name` tag entries carry their `coverage_metadata`
            block (the ranked canon candidates with embedding scores,
            the LLM's confidence label, and the evidence sentence) plus
            a derived `selected: true|false` marker per candidate.
            When False (default), `coverage_metadata` is stripped to keep
            the payload small.

    Pure function — same inputs always yield the same dict.
    """
    if not isinstance(tagged, dict):
        raise TypeError(f"tagged must be dict, got {type(tagged).__name__}")

    project_tags = _normalize_project_tags(
        tagged.get("project_tags") or {},
        tenant_id=tagged.get("tenant_id"),
        survey_no=tagged.get("survey_no"),
    )
    raw_question_tags = tagged.get("question_tags") or []

    # Questions: passthrough shape, but elevate `tags` so consumers don't have
    # to peek under another key.
    questions: list[dict] = []
    for q in raw_question_tags:
        if not isinstance(q, dict):
            continue
        questions.append({
            "question_id":        q.get("question_id"),
            "question_no":        q.get("question_no"),
            "question_text":      q.get("question_text") or q.get("question_title_preview"),
            "rs_type":            q.get("rs_type", 0),
            "is_custom_metric":   bool(q.get("is_custom_metric", False)),
            "is_content_message": bool(q.get("is_content_message", False)),
            "tags":               _shape_question_tags(q.get("tags") or {},
                                                       include_journey_candidates),
        })

    project_type = get_tag_value(project_tags, "project_type")
    survey_journey = _build_survey_journey(project_type, questions)

    out: dict = {
        "tenant_id":      tagged.get("tenant_id"),
        "survey_no":      tagged.get("survey_no"),
        "zarca_id":       tagged.get("zarca_id"),
        "survey_name":    tagged.get("survey_name"),
        "schema_version": SCHEMA_VERSION,
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_tags":   project_tags,
    }
    if survey_journey is not None:
        out["survey_journey"] = survey_journey
    out["questions"] = questions
    out["metadata"] = tagged.get("metadata") or {}
    return out


def _shape_question_tags(
    tags: dict,
    include_journey_candidates: bool,
) -> dict:
    """Pass through every tag; on the journey dimensions, either strip
    `coverage_metadata` (default) or expose it with a `selected` flag per
    candidate so consumers can render "why this stage" without re-deriving it.
    """
    out: dict = {}
    for dim, entry in tags.items():
        if not isinstance(entry, dict):
            out[dim] = entry
            continue
        if dim not in _COVERAGE_DIMS:
            out[dim] = entry
            continue

        if not include_journey_candidates:
            # Strip coverage_metadata to keep the default response lean
            shaped = {k: v for k, v in entry.items() if k != "coverage_metadata"}
            out[dim] = shaped
            continue

        # include_journey_candidates=True: keep coverage_metadata, decorate
        # the candidate list with `selected` for the picked stage so the
        # consumer doesn't have to recompute the match.
        cov = entry.get("coverage_metadata")
        if not isinstance(cov, dict):
            out[dim] = entry
            continue
        candidates = cov.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            out[dim] = entry
            continue

        # Prefer the leaf id: with a two-level journey several candidates can
        # share a stage name, and on the `sub_stage_name` dimension the tag
        # value is the sub-stage, so a name comparison marks the wrong row (or
        # none at all). Fall back to the name for artifacts written before
        # leaf ids existed.
        picked = entry.get("value")
        picked_leaf = cov.get("leaf_id")
        decorated = []
        for c in candidates:
            if not isinstance(c, dict):
                decorated.append(c)
                continue
            if picked_leaf and c.get("leaf_id"):
                selected = c.get("leaf_id") == picked_leaf
            else:
                cand_name = c.get("stage_name") or c.get("name")
                selected = cand_name == picked
            decorated.append({**c, "selected": selected})

        shaped_cov = {**cov, "candidates": decorated}
        out[dim] = {**entry, "coverage_metadata": shaped_cov}
    return out


def _normalize_project_tags(
    project_tags: dict,
    *,
    tenant_id: int | None = None,
    survey_no: int | None = None,
) -> dict:
    """V6 backward-compat: pre-V6 tagged_output.json files carry `category`
    instead of `project_type`. Rename it on-the-fly so the endpoint contract
    is consistent regardless of when the file was tagged.

    Logs a warning so operators know the underlying file is stale and should
    be re-tagged to migrate cleanly.
    """
    if "category" in project_tags and "project_type" not in project_tags:
        logger.warning(
            "survey_view_legacy_category_key",
            extra={
                "tenant_id": tenant_id,
                "survey_no": survey_no,
                "hint": "tagged_output.json predates V6 rename. "
                        "POST /api/retag/{tenant}/{survey} to migrate.",
            },
        )
        out = dict(project_tags)
        out["project_type"] = out.pop("category")
        return out
    return project_tags


def _build_survey_journey(project_type: str | None, questions: list[dict]) -> dict | None:
    """Derive {journey_type, stages_touched, sub_stages_touched} from question
    tags. Returns None when the survey isn't CX/EX (journey is N/A).

    Stage/sub-stage order = first-appearance in `questions` order. Deduped.
    Skipped tags (absent from `tags`) contribute nothing.
    """
    if project_type not in _JOURNEY_TYPES:
        return None

    stages = list(_dedup(_iter_tag_values(questions, "journey_stage")))
    sub_stages = list(_dedup(_iter_tag_values(questions, "sub_stage_name")))

    return {
        "journey_type":       project_type,
        "stages_touched":     stages,
        "sub_stages_touched": sub_stages,
    }


def _iter_tag_values(questions: list[dict], dim: str) -> Iterable[str]:
    for q in questions:
        if q.get("is_content_message"):
            continue
        val = get_tag_value(q.get("tags") or {}, dim)
        if isinstance(val, str) and val.strip():
            yield val.strip()


def _dedup(values: Iterable[str]) -> Iterable[str]:
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            yield v
