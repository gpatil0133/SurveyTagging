"""Parse and validate LLM responses against the taxonomy.

V2: supports user_defined bypass, multi-label list values, and industry-specific
journey_stage fuzzy matching.
"""

from __future__ import annotations

import logging

from config_loaders.industry_stages import IndustryStagesRegistry
from models.taxonomy import TaxonomyRegistry

logger = logging.getLogger(__name__)


# V7.1: the output keys each LLM call may attach a per-dimension `why` line to.
# Anything else in the model's `why` map is dropped — a hallucinated key would
# otherwise surface as an explanation for a tag that call never assigned.
_PROJECT_WHY_KEYS = frozenset({
    "relationship_type", "project_purpose", "industry_vertical",
    "audience_type_refined", "survey_sub_type", "dashboard_names",
})
_QUESTION_WHY_KEYS = frozenset({
    "topic_theme", "role_intent_refined", "respondent_sensitivity",
    "flow_respondent_experience", "flow_reusability", "visualization_type",
    "dashboard_names", "display_role",
})

# A `why` line is one sentence by contract. Truncate rather than reject so a
# chatty model still explains itself.
_WHY_MAX_CHARS = 300


def _clean_why(raw, allowed_keys: frozenset[str]) -> dict[str, str]:
    """Normalize the model's per-dimension `why` map: keep known keys with
    non-empty string values, collapse whitespace, bound the length."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, line in raw.items():
        if key not in allowed_keys or not isinstance(line, str):
            continue
        collapsed = " ".join(line.split())
        if collapsed:
            out[key] = collapsed[:_WHY_MAX_CHARS]
    return out


class ResponseParser:
    """Validates LLM output against taxonomy and extracts structured tags."""

    def __init__(
        self,
        taxonomy: TaxonomyRegistry,
        industry_stages: IndustryStagesRegistry | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        # PARKED: only the removed journey fallback ever read this. Kept on the
        # constructor so existing call sites (orchestrator, tests) still build.
        self.industry_stages = industry_stages

    @staticmethod
    def _note_normalization(
        why: dict[str, str], field: str, model_value: str, stored_value: str | None
    ) -> None:
        """Append a note to the `why` line when validation rewrote the model's
        answer. Without this the stored explanation argues for a value the tag
        no longer holds — the single most misleading thing an audit trail can do.
        """
        if stored_value == model_value:
            return
        if stored_value:
            note = (f'Model answered "{model_value}", which is not an allowed value; '
                    f"snapped to the nearest one.")
        else:
            note = (f'Model answered "{model_value}", which matched no allowed value; '
                    f"the tag was dropped.")
        why[field] = f"{why[field]} — {note}" if why.get(field) else note

    # ---------- Project-level ----------

    def parse_project_response(self, data: dict) -> dict:
        """Parse and validate project-level LLM response."""
        logger.debug("parse_project_response_start",
                     extra={"input_keys": list(data.keys())[:20]})
        result: dict = {}

        # Scalar fields
        scalar_map = {
            "relationship_type": "relationship_type",
            "project_purpose": "project_purpose",
            "industry_vertical": "industry_vertical",
            "audience_type_refined": "audience_type",
            "survey_sub_type": "survey_sub_type",
        }
        why = _clean_why(data.get("why"), _PROJECT_WHY_KEYS)

        for field, dimension in scalar_map.items():
            value = data.get(field)
            if value:
                validated = self._validate_scalar(dimension, value)
                result[field] = validated
                self._note_normalization(why, field, value, validated)
            else:
                result[field] = None
                why.pop(field, None)  # explained a dimension it never set

        # Multi-label: dashboard_names → dashboard_routing
        dashboard_names = data.get("dashboard_names")
        if dashboard_names and isinstance(dashboard_names, list):
            result["dashboard_names"] = [
                self._validate_list_item("dashboard_routing", v) for v in dashboard_names
            ]
            result["dashboard_names"] = [v for v in result["dashboard_names"] if v]
        else:
            result["dashboard_names"] = None
            why.pop("dashboard_names", None)

        result["why"] = why
        result["reasoning"] = data.get("reasoning", "")
        logger.debug(
            "parse_project_response_done",
            extra={"scalar_fields": {k: result.get(k) for k in scalar_map},
                   "dashboard_names": result.get("dashboard_names")},
        )
        return result

    # ---------- Question-level ----------

    def parse_question_response(
        self, data: dict, industry: str | None = None,
        project_type: str | None = None,
        candidates_by_qid: dict[int, list[dict]] | None = None,  # per-question candidate lists
    ) -> list[dict]:
        """Parse and validate question-level LLM response.

        V8: the `journey` block is resolved through `candidates_by_qid` — the
        same ranking the prompt showed the model — so both journey tag values
        come from the tenant's profile rather than from model text. A question
        with no candidates has no journey namespace and gets no journey tags.
        """
        questions = data.get("questions", [])
        logger.debug("parse_question_response_start",
                     extra={"question_count": len(questions) if isinstance(questions, list) else 0,
                            "candidate_qids": len(candidates_by_qid or {}),
                            "industry": industry, "project_type": project_type})
        if not isinstance(questions, list):
            logger.warning("llm_questions_not_list")
            return []

        candidates_by_qid = candidates_by_qid or {}

        results = []
        for q_data in questions:
            if not isinstance(q_data, dict):
                continue
            parsed: dict = {"id": q_data.get("id")}

            scalar_map = {
                "topic_theme": "topic_theme",
                "respondent_sensitivity": "respondent_sensitivity",
                "flow_respondent_experience": "flow_respondent_experience",
                "flow_reusability": "flow_reusability",
                "visualization_type": "visualization_type",
                "display_role": "display_role",
            }
            why = _clean_why(q_data.get("why"), _QUESTION_WHY_KEYS)

            for field, dimension in scalar_map.items():
                value = q_data.get(field)
                if value:
                    validated = self._validate_scalar(dimension, value)
                    parsed[field] = validated
                    self._note_normalization(why, field, value, validated)
                else:
                    why.pop(field, None)  # explained a dimension it never set

            # ---------- Journey block (leaf selected from this question's candidates) ----------
            qid = q_data.get("id")
            qid_int = int(qid) if isinstance(qid, int) or (isinstance(qid, str) and qid.isdigit()) else None
            self._parse_journey_for_question(
                q_data=q_data, parsed=parsed,
                candidates=candidates_by_qid.get(qid_int) if qid_int is not None else None,
            )

            # dashboard_names multi-label
            dashboard_names = q_data.get("dashboard_names")
            if dashboard_names and isinstance(dashboard_names, list):
                cleaned = [self._validate_list_item("dashboard_placement", v)
                           for v in dashboard_names]
                parsed["dashboard_names"] = [v for v in cleaned if v]
            else:
                why.pop("dashboard_names", None)

            # Role intent refinement (preserved from v1)
            parsed["role_intent_refined"] = q_data.get("role_intent_refined")
            if not parsed["role_intent_refined"]:
                why.pop("role_intent_refined", None)

            # V7.1: one rationale line PER dimension, so a reader inspecting a
            # single tag gets the signal behind that tag rather than a blob
            # covering the other seven.
            parsed["why"] = why

            # V7: the question-level summary. Still carried, now as the fallback
            # for any dimension the model gave no `why` line for — and for
            # cached responses from prompt version < 7.1, which have no `why`.
            reasoning = q_data.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                parsed["reasoning"] = reasoning.strip()

            results.append(parsed)

        logger.debug("parse_question_response_done",
                     extra={"parsed_questions": len(results)})
        return results

    # ---------- Journey block parser ----------

    def _parse_journey_for_question(
        self,
        *,
        q_data: dict,
        parsed: dict,
        candidates: list[dict] | None,
    ) -> None:
        """Resolve the LLM's `journey` block into stage + sub-stage tag values.

        V8: the model returns a `leaf_id` selected from this question's own
        candidate list, and both values are read off that candidate. Nothing the
        model types becomes a tag value, so a stage can no longer be invented,
        misspelled, or fuzzy-matched into the wrong place — the failure mode the
        canon path spent a validation ladder defending against.

        An unresolvable pick falls back to the top-ranked candidate as
        `low_confidence_assigned` rather than being dropped, since the embedding
        ranking is independent evidence that the question belongs somewhere.
        A model that declined to place the question (`journey: null`) is left
        alone — the caller marks it skipped with the reason.

        Writes nothing when there is no assignment to make.
        """
        journey = q_data.get("journey")
        if not isinstance(journey, dict) or not journey:
            return
        if not candidates:
            # The model placed a question we gave it no candidates for. There is
            # no namespace to resolve against, so there is no safe value.
            logger.warning("journey_assigned_without_candidates",
                           extra={"qid": q_data.get("id")})
            return

        by_leaf = {c.get("leaf_id"): c for c in candidates if c.get("leaf_id")}
        leaf_id = journey.get("leaf_id")
        leaf_id = leaf_id.strip() if isinstance(leaf_id, str) else None

        status = "assigned"
        confidence = str(journey.get("confidence") or "medium").lower().strip()
        evidence = journey.get("evidence")

        chosen = by_leaf.get(leaf_id) if leaf_id else None
        if chosen is None:
            # Tolerate a model that named the moment instead of copying the id,
            # then fall back to the ranking.
            chosen = self._match_candidate_by_name(journey, candidates)
            if chosen is None:
                chosen = candidates[0]
                logger.warning(
                    "journey_leaf_unresolved",
                    extra={"qid": q_data.get("id"), "llm_leaf_id": leaf_id,
                           "fallback": chosen.get("leaf_id")},
                )
                evidence = (evidence + " | " if evidence else "") + (
                    "LLM returned an unknown leaf_id; defaulted to the top-ranked "
                    "candidate."
                )
            status = "low_confidence_assigned"

        stage_value = chosen.get("stage_name")
        if not stage_value:
            return
        sub_value = chosen.get("sub_stage_name") or None

        # ---------- Confidence -> status escalation ----------
        if confidence in ("low", "none") and status == "assigned":
            status = "low_confidence_assigned"

        parsed["journey_stage"] = stage_value
        parsed["journey_leaf_id"] = chosen.get("leaf_id")
        # None is a real answer here: a one-level journey (EX lifecycle) has no
        # sub-stage, and inventing one is what produced metric names in the
        # sub_stage_name column under the canon path.
        parsed["sub_stage_name"] = sub_value
        parsed["journey_status"] = status
        parsed["journey_confidence"] = (
            confidence if confidence in ("high", "medium", "low", "none") else "medium"
        )
        if evidence:
            parsed["journey_evidence"] = evidence
        # `leaf_id` rides along so a reader can tell WHICH candidate was picked.
        # Name alone no longer identifies one: a two-level journey legitimately
        # has several moments under the same stage.
        parsed["journey_candidates"] = [
            {"leaf_id": c.get("leaf_id"), "name": c.get("stage_name"),
             "sub_stage": c.get("sub_stage_name"), "score": c.get("score")}
            for c in candidates
        ]

    @staticmethod
    def _match_candidate_by_name(journey: dict, candidates: list[dict]) -> dict | None:
        """Second chance for a model that wrote names instead of the leaf_id.

        Matches on the (stage, sub_stage) pair when both were given, else on
        whichever single name was. Case-insensitive; no fuzzy matching — a near
        miss goes to the ranking fallback, which is honest about being one.
        """
        stage = journey.get("stage_name")
        sub = journey.get("sub_stage_name")
        stage = stage.strip().lower() if isinstance(stage, str) else ""
        sub = sub.strip().lower() if isinstance(sub, str) else ""
        if not stage and not sub:
            return None

        for c in candidates:
            c_stage = str(c.get("stage_name") or "").strip().lower()
            c_sub = str(c.get("sub_stage_name") or "").strip().lower()
            if stage and sub:
                if c_stage == stage and c_sub == sub:
                    return c
            elif stage and c_stage == stage:
                return c
            elif sub and c_sub == sub:
                return c
        return None

    # ---------- Validation helpers ----------

    def _validate_scalar(self, dimension: str, value: str) -> str | None:
        """Validate a single value against a scalar dimension."""
        dim = self.taxonomy.get_dimension(dimension)
        if dim is None:
            logger.warning("unknown_dimension", extra={"dimension": dimension})
            return value
        if dim.user_defined:
            return value  # Accept free-text as-is
        errors = self.taxonomy.validate_value(dimension, value)
        if errors:
            closest = self._find_closest(value, dimension)
            if closest is None:
                logger.warning("no_close_match",
                               extra={"value": value, "dimension": dimension})
            return closest
        return value

    def _validate_list_item(self, dimension: str, value: str) -> str | None:
        """Validate one item of a multi-label dimension."""
        dim = self.taxonomy.get_dimension(dimension)
        if dim is None:
            return value
        if dim.user_defined:
            # For canonical_values-backed user_defined dims, keep value as-is
            # (prompt biases LLM toward canonical names)
            return value if value else None
        errors = dim.validate_tag(value)
        if errors:
            return self._find_closest(value, dimension)
        return value

    def _find_closest(self, value: str, dimension: str) -> str | None:
        """Find the closest allowed value for a dimension (simple substring match)."""
        dim = self.taxonomy.get_dimension(dimension)
        if dim is None or not dim.allowed_values:
            # user_defined or unknown → accept as-is (F4/F5 fix)
            return value if value else None

        allowed = dim.allowed_values
        value_lower = value.lower().strip()
        for av in allowed:
            if av.lower() == value_lower:
                return av
        for av in allowed:
            if value_lower in av.lower() or av.lower() in value_lower:
                return av
        logger.warning("no_close_match", extra={"value": value, "dimension": dimension})
        return None

