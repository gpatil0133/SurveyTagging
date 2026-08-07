"""Parse and validate LLM responses against the taxonomy.

V2: supports user_defined bypass, multi-label list values, and industry-specific
journey_stage fuzzy matching.
"""

from __future__ import annotations

import logging

from config_loaders.industry_stages import IndustryStagesRegistry
from models.taxonomy import TaxonomyRegistry

logger = logging.getLogger(__name__)


class ResponseParser:
    """Validates LLM output against taxonomy and extracts structured tags."""

    def __init__(
        self,
        taxonomy: TaxonomyRegistry,
        industry_stages: IndustryStagesRegistry | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.industry_stages = industry_stages

    @staticmethod
    def _is_word_count_ok(label: str, *, min_w: int = 2, max_w: int = 6) -> bool:
        n = len(label.split())
        return min_w <= n <= max_w

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
        for field, dimension in scalar_map.items():
            value = data.get(field)
            if value:
                result[field] = self._validate_scalar(dimension, value)
            else:
                result[field] = None

        # Multi-label: dashboard_names → dashboard_routing
        dashboard_names = data.get("dashboard_names")
        if dashboard_names and isinstance(dashboard_names, list):
            result["dashboard_names"] = [
                self._validate_list_item("dashboard_routing", v) for v in dashboard_names
            ]
            result["dashboard_names"] = [v for v in result["dashboard_names"] if v]
        else:
            result["dashboard_names"] = None

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
        canon=None,                # V5: TenantCanon | None
        candidates_by_qid: dict[int, list[dict]] | None = None,  # V5: per-question candidate lists
    ) -> list[dict]:
        """Parse and validate question-level LLM response.

        V5: when `canon` is provided, the LLM's atomic `journey` block is
        parsed and validated against the canon namespace; out-of-canon stage
        names are downgraded to `low_confidence_assigned` keyed to the top-1
        candidate so no question is silently dropped.

        When `canon` is None (legacy tenants), falls back to the v4.x
        `journey_stage` + `sub_stage_name` parsing path.
        """
        questions = data.get("questions", [])
        logger.debug("parse_question_response_start",
                     extra={"question_count": len(questions) if isinstance(questions, list) else 0,
                            "has_canon": canon is not None,
                            "industry": industry, "project_type": project_type})
        if not isinstance(questions, list):
            logger.warning("llm_questions_not_list")
            return []

        canon_names: set[str] = set()
        canon_lower: dict[str, str] = {}
        if canon is not None:
            canon_names = {s.name for s in canon.stages}
            canon_lower = {s.name.lower(): s.name for s in canon.stages}

        # Legacy industry-stage list (only used when canon is None)
        industry_stages_list: list[str] = []
        if canon is None and self.industry_stages is not None:
            industry_stages_list = self.industry_stages.get_stages(industry, project_type=project_type)

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
            for field, dimension in scalar_map.items():
                value = q_data.get(field)
                if value:
                    parsed[field] = self._validate_scalar(dimension, value)

            # ---------- Journey block (V5 atomic OR v4 legacy) ----------
            qid = q_data.get("id")
            qid_int = int(qid) if isinstance(qid, int) or (isinstance(qid, str) and qid.isdigit()) else None
            self._parse_journey_for_question(
                q_data=q_data, parsed=parsed,
                canon_names=canon_names, canon_lower=canon_lower,
                candidates=candidates_by_qid.get(qid_int) if qid_int is not None else None,
                legacy_stage_list=industry_stages_list,
                use_v5=canon is not None,
            )

            # dashboard_names multi-label
            dashboard_names = q_data.get("dashboard_names")
            if dashboard_names and isinstance(dashboard_names, list):
                cleaned = [self._validate_list_item("dashboard_placement", v)
                           for v in dashboard_names]
                parsed["dashboard_names"] = [v for v in cleaned if v]

            # Role intent refinement (preserved from v1)
            parsed["role_intent_refined"] = q_data.get("role_intent_refined")
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
        canon_names: set[str],
        canon_lower: dict[str, str],
        candidates: list[dict] | None,
        legacy_stage_list: list[str],
        use_v5: bool,
    ) -> None:
        """Parse the LLM's `journey` block (V5) or legacy fields (v4) and write
        into `parsed`. Always emits non-null `sub_stage_name` whenever the
        stage was assigned. Surfaces `journey_status` ∈
        {"assigned", "low_confidence_assigned"} so the orchestrator can map it
        onto TagResult.status.
        """
        journey = q_data.get("journey")
        # Legacy v4 fallback path: pick up old fields if the new block is absent.
        legacy_stage = q_data.get("journey_stage")
        legacy_sub = q_data.get("sub_stage_name")

        stage_value: str | None = None
        sub_value: str | None = None
        confidence: str = "high"
        evidence: str | None = None

        if isinstance(journey, dict) and journey:
            stage_raw = journey.get("stage_name")
            sub_raw = journey.get("sub_stage_name")
            confidence = str(journey.get("confidence") or "medium").lower().strip()
            evidence = journey.get("evidence")
            if isinstance(stage_raw, str) and stage_raw and stage_raw.lower() != "null":
                stage_value = stage_raw.strip()
            if isinstance(sub_raw, str) and sub_raw and sub_raw.lower() != "null":
                sub_value = sub_raw.strip()
        elif legacy_stage and legacy_stage != "null":
            # v4 fallback (used during transition before all surveys re-tagged)
            stage_value = str(legacy_stage).strip()
            if isinstance(legacy_sub, str) and legacy_sub.strip() and legacy_sub.strip().lower() != "null":
                sub_value = legacy_sub.strip()

        if not stage_value:
            # Nothing to assign. journey is null; downstream tagger will still
            # emit pending_llm → skipped via metric eligibility check.
            return

        # ---------- Validate stage_name against canon (V5) or legacy list ----------
        status = "assigned"
        if use_v5 and canon_names:
            if stage_value in canon_names:
                pass  # exact match
            elif stage_value.lower() in canon_lower:
                stage_value = canon_lower[stage_value.lower()]  # case fix
            else:
                # Out-of-canon: downgrade to top-1 candidate, low confidence
                fallback = (candidates or [{}])[0].get("stage_name") if candidates else None
                if fallback:
                    logger.warning(
                        "journey_stage_out_of_canon",
                        extra={"qid": q_data.get("id"),
                               "llm_value": stage_value, "fallback": fallback},
                    )
                    stage_value = fallback
                    status = "low_confidence_assigned"
                    confidence = "low"
                    evidence = (evidence + " | " if evidence else "") + \
                        f"LLM emitted out-of-canon stage; defaulted to top-1 candidate"
                else:
                    # No candidates and out-of-canon — drop the assignment.
                    logger.warning(
                        "journey_stage_no_canon_match_no_candidates",
                        extra={"qid": q_data.get("id"), "llm_value": stage_value},
                    )
                    return
        elif not use_v5 and legacy_stage_list:
            # Legacy v4 fuzzy match against industry list
            stage_value = self._find_closest_in_list(stage_value, legacy_stage_list) or stage_value

        # ---------- Validate sub_stage_name; deterministic fallback when missing ----------
        if not sub_value or not self._is_word_count_ok(sub_value, max_w=6):
            sub_value = f"Other {stage_value}"
            if status == "assigned":
                status = "low_confidence_assigned"
                if not evidence:
                    evidence = "LLM omitted sub_stage_name; defaulted from stage."

        # ---------- Confidence → status escalation ----------
        if confidence in ("low", "none") and status == "assigned":
            status = "low_confidence_assigned"

        parsed["journey_stage"] = stage_value
        parsed["sub_stage_name"] = sub_value
        parsed["journey_status"] = status
        parsed["journey_confidence"] = confidence if confidence in ("high", "medium", "low", "none") else "medium"
        if evidence:
            parsed["journey_evidence"] = evidence
        if candidates:
            parsed["journey_candidates"] = [
                {"name": c.get("stage_name"), "score": c.get("score")}
                for c in candidates
            ]

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

    def _find_closest_in_list(self, value: str, allowed: list[str]) -> str | None:
        """Fuzzy match against an explicit list (used for industry-specific stages)."""
        if not allowed:
            return value
        value_lower = value.lower().strip()
        for av in allowed:
            if av.lower() == value_lower:
                return av
        for av in allowed:
            if value_lower in av.lower() or av.lower() in value_lower:
                return av
        logger.warning("journey_stage_no_match",
                       extra={"value": value, "allowed": allowed})
        return None
