"""Tenant canon derivation — single LLM call per tenant per journey type.

Replaces the old `derive_tenant_stages` flow which never read agent stage data.
This module:
  - Aggregates raw stages from `tenant_profile.cx_journeys` (or `ex_lifecycle_stages`).
  - Gates: agent_canon (trust agent) / agent_blended (LLM merge) / industry_template (no agent).
  - Calls LLM ONCE to canonicalize: dedupe synonyms, order chronologically, attach
    synonyms list per stage. Skipped entirely in `industry_template` mode.
  - Returns a `TenantCanon` Pydantic model suitable for persisting via
    `projections.tenant_canon_io.save_tenant_canon`.

Public surface:
  - `build_tenant_canon(...)` — async, the main entry point.
  - `CANON_PROMPT_VERSION` constant — bump on prompt schema changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Literal

from llm.client import LLMClient
from models.tenant_canon import CanonStage, JourneyType, TenantCanon
from models.tenant_profile import TenantProfile

logger = logging.getLogger(__name__)


# Bump when the canonicalization prompt or its output schema changes.
CANON_PROMPT_VERSION = "1.0"

# Constants for gating
_MIN_RAW_STAGES_FOR_AGENT = 4
_MIN_RAW_STAGES_FOR_BLEND = 3
_TRUSTED_CONFIDENCE = "High"
_RICH_FIELD_RATIO = 0.70  # ≥ this fraction of stages must have description AND customer_goal


# ---------- System preamble (cached on Anthropic) ----------

_CANON_PREAMBLE = """You are a senior CX/EX journey architect canonicalizing a tenant's journey stages.

You receive:
  1. Raw agent-derived stages: stage_name, description, customer_goal, source_journey.
     Multiple journeys can contain the same logical stage under different names.
  2. The industry's reference stage template (sanity check, not a target).
  3. A canonicalization mode:
       "agent_canon"   — trust the agent. Merge ONLY exact near-duplicates; keep agent stage names verbatim.
       "agent_blended" — agent is sparse or lower-confidence. Combine agent stages with the
                         industry template, preferring agent names where they cover the same touchpoint.

Rules:
  - PRESERVE granularity. If the agent emitted "OPD Registration" and "Pre-Admission & Cost Estimate"
    as DISTINCT stages, they MUST remain distinct. Never collapse evidence-grounded agent stages
    into a generic name.
  - Order stages chronologically along the journey (earliest customer/employee touchpoint first).
  - For each canonical stage produce:
      name (verbatim from agent in agent_canon; chosen from template in industry_template-blend cases)
      description (one sentence, business-specific, NOT generic)
      customer_goal (one short clause; copy or merge agent goals)
      synonyms (3-8 short phrases — what customers/employees might use to refer to this moment;
                e.g. "checkout", "billing & discharge", "leaving the hospital")
      source_journeys (list of agent journey names this stage came from; empty for template-only stages)
      industry_template_match (nearest reference-template stage name, or null if no clear match)

Return JSON only. No preamble, no commentary."""


_CANON_SCHEMA = """Output schema:
{
  "journey_name": "<single phrase, e.g. 'Patient Journey'>",
  "stages": [
    {
      "name": "<exact agent name in agent_canon mode>",
      "description": "<one sentence>",
      "customer_goal": "<one short clause>",
      "synonyms": ["<3-8 short phrases>"],
      "source_journeys": ["<agent journey names>"],
      "industry_template_match": "<reference-template stage name OR null>"
    },
    ...
  ]
}"""


# ---------- Public API ----------


def compute_canon_input_hash(
    *,
    tenant_profile: TenantProfile | None,
    journey_type: JourneyType,
    industry_stage_template: list[dict],
) -> str:
    """Compute the canon `input_hash` for the current inputs WITHOUT building.

    Mirrors the aggregate → dedup → gate → hash sequence inside
    `build_tenant_canon` exactly, so callers can compare against a persisted
    canon's `input_hash` to decide whether a rebuild is warranted (e.g. after a
    tenant profile is fetched or changes). Keep this in lockstep with
    `build_tenant_canon`.
    """
    raw = _aggregate_raw_stages(tenant_profile, journey_type)
    deduped = _light_dedup(raw)
    mode = _gate(tenant_profile, journey_type, deduped)
    return _input_hash(deduped, industry_stage_template, journey_type, mode)


async def build_tenant_canon(
    *,
    llm: LLMClient | None,
    tenant_profile: TenantProfile | None,
    journey_type: JourneyType,
    industry_stage_template: list[dict],
    industry: str = "",
    corporate_purpose: str = "",
    tenant_id: int,
    force: bool = False,
) -> TenantCanon:
    """Build the tenant canon for one journey type.

    Returns a fully-formed `TenantCanon` whose `source` indicates which path
    was taken. Persisting is the caller's responsibility.

    Args:
        llm: LLMClient or None. None → industry_template mode forced.
        tenant_profile: loaded TenantProfile or None.
        journey_type: "CX" or "EX".
        industry_stage_template: list of {"name", "description"} from the
            industry template (registry.get_stages mapped via _STAGE_DESCRIPTIONS).
        industry: canonical industry vertical name.
        corporate_purpose: tenant mission/description.
        tenant_id: required for the resulting canon.
        force: bypass LLM cache for the canonicalization call.
    """
    raw = _aggregate_raw_stages(tenant_profile, journey_type)
    deduped = _light_dedup(raw)
    mode = _gate(tenant_profile, journey_type, deduped)
    input_hash = _input_hash(deduped, industry_stage_template, journey_type, mode)

    logger.info(
        "tenant_canon_build_start",
        extra={
            "tenant_id": tenant_id,
            "journey_type": journey_type,
            "raw_count": len(raw),
            "deduped_count": len(deduped),
            "mode": mode,
        },
    )

    if mode == "industry_template" or llm is None:
        canon_stages = _stages_from_template(industry_stage_template)
        return TenantCanon(
            tenant_id=tenant_id,
            journey_type=journey_type,
            journey_name=_default_journey_name(journey_type, industry),
            industry=industry,
            source="industry_template",
            derived_at=_now(),
            confidence="synthesized",
            stages=canon_stages,
            input_hash=input_hash,
        )

    canon_dict = await _llm_canonicalize(
        llm=llm, mode=mode, deduped=deduped, industry_template=industry_stage_template,
        industry=industry, corporate_purpose=corporate_purpose,
        journey_type=journey_type, tenant_id=tenant_id, input_hash=input_hash, force=force,
    )

    if canon_dict is None:
        # LLM failed or response invalid; fall back to using the raw agent
        # stages directly (preserves agent data, just no synonyms).
        if mode == "agent_canon" and deduped:
            canon_stages = _stages_from_deduped(deduped)
            return TenantCanon(
                tenant_id=tenant_id, journey_type=journey_type,
                journey_name=_default_journey_name(journey_type, industry),
                industry=industry, source="agent_canon",
                derived_at=_now(), confidence=(tenant_profile.cx_confidence if tenant_profile and journey_type == "CX" else
                                               tenant_profile.ex_confidence if tenant_profile else "synthesized"),
                stages=canon_stages, input_hash=input_hash,
            )
        # Blended mode failure → degrade to industry template.
        canon_stages = _stages_from_template(industry_stage_template)
        return TenantCanon(
            tenant_id=tenant_id, journey_type=journey_type,
            journey_name=_default_journey_name(journey_type, industry),
            industry=industry, source="industry_template",
            derived_at=_now(), confidence="synthesized",
            stages=canon_stages, input_hash=input_hash,
        )

    return TenantCanon(
        tenant_id=tenant_id, journey_type=journey_type,
        journey_name=canon_dict["journey_name"],
        industry=industry,
        source=mode,  # "agent_canon" or "agent_blended"
        derived_at=_now(),
        confidence=(tenant_profile.cx_confidence if tenant_profile and journey_type == "CX" else
                    tenant_profile.ex_confidence if tenant_profile else "synthesized"),
        stages=canon_dict["stages"],
        input_hash=input_hash,
    )


# ---------- Aggregation + dedup ----------


def _aggregate_raw_stages(
    tenant_profile: TenantProfile | None, journey_type: JourneyType,
) -> list[dict]:
    """Pass 0: collect (journey, stage_name, description, customer_goal) from agent."""
    if tenant_profile is None:
        return []
    if journey_type == "CX":
        if not tenant_profile.has_cx:
            return []
        records: list[dict] = []
        for j in tenant_profile.cx_journeys:
            jname = str(j.get("journey_name") or "")
            for s in (j.get("stages") or []):
                if not isinstance(s, dict):
                    continue
                name = str(s.get("stage_name") or "").strip()
                if not name:
                    continue
                records.append({
                    "journey_name": jname,
                    "stage_name": name,
                    "description": str(s.get("description") or "").strip(),
                    "customer_goal": str(s.get("customer_goal") or "").strip(),
                })
        return records

    # EX
    if not tenant_profile.has_ex:
        return []
    records = []
    for s in tenant_profile.ex_lifecycle_stages:
        if not isinstance(s, dict):
            continue
        name = str(s.get("stage_name") or "").strip()
        if not name:
            continue
        records.append({
            "journey_name": "Employee Lifecycle",
            "stage_name": name,
            "description": str(s.get("description") or "").strip(),
            "customer_goal": str(s.get("employee_goal") or s.get("customer_goal") or "").strip(),
        })
    return records


def _normalize_for_dedup(name: str) -> str:
    """Lowercase + replace non-alphanumeric with spaces + collapse whitespace.

    "OPD Registration", "OPD-Registration", "opd_registration" all map to the
    same key so light dedup collapses them.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()


def _light_dedup(raw: list[dict]) -> list[dict]:
    """Pass 1: collapse exact (case/punctuation-insensitive) duplicate stage names.

    When dupes exist, merge their `journey_name` into a list and keep the
    longer description / customer_goal.
    """
    out: dict[str, dict] = {}
    for rec in raw:
        key = _normalize_for_dedup(rec["stage_name"])
        if not key:
            continue
        if key not in out:
            out[key] = {
                "stage_name": rec["stage_name"],
                "description": rec["description"],
                "customer_goal": rec["customer_goal"],
                "source_journeys": [rec["journey_name"]] if rec["journey_name"] else [],
            }
        else:
            entry = out[key]
            if rec["journey_name"] and rec["journey_name"] not in entry["source_journeys"]:
                entry["source_journeys"].append(rec["journey_name"])
            if len(rec["description"]) > len(entry["description"]):
                entry["description"] = rec["description"]
            if len(rec["customer_goal"]) > len(entry["customer_goal"]):
                entry["customer_goal"] = rec["customer_goal"]
    return list(out.values())


# ---------- Gating ----------


CanonMode = Literal["agent_canon", "agent_blended", "industry_template"]


def _gate(
    tenant_profile: TenantProfile | None,
    journey_type: JourneyType,
    deduped: list[dict],
) -> CanonMode:
    """Pass 2: choose canonicalization mode."""
    if tenant_profile is None:
        return "industry_template"
    if journey_type == "CX" and not tenant_profile.has_cx:
        return "industry_template"
    if journey_type == "EX" and not tenant_profile.has_ex:
        return "industry_template"
    if len(deduped) < _MIN_RAW_STAGES_FOR_BLEND:
        return "industry_template"

    confidence = (tenant_profile.cx_confidence if journey_type == "CX"
                  else tenant_profile.ex_confidence)

    rich = sum(1 for s in deduped if s["description"] and s["customer_goal"])
    rich_ratio = rich / len(deduped) if deduped else 0.0

    if (
        confidence == _TRUSTED_CONFIDENCE
        and len(deduped) >= _MIN_RAW_STAGES_FOR_AGENT
        and rich_ratio >= _RICH_FIELD_RATIO
    ):
        return "agent_canon"

    return "agent_blended"


# ---------- LLM call ----------


async def _llm_canonicalize(
    *,
    llm: LLMClient,
    mode: CanonMode,
    deduped: list[dict],
    industry_template: list[dict],
    industry: str,
    corporate_purpose: str,
    journey_type: JourneyType,
    tenant_id: int,
    input_hash: str,
    force: bool,
) -> dict | None:
    """Pass 3: LLM canonicalization. Returns {journey_name, stages: [CanonStage]}."""
    cache_key = f"canon_{tenant_id}_{journey_type.lower()}_{input_hash}"
    call_type = f"tenant_canon_{journey_type.lower()}"

    if force and llm.cache:
        llm.cache.invalidate(cache_key)

    user_prompt = _build_user_prompt(
        mode=mode, deduped=deduped, industry_template=industry_template,
        industry=industry, corporate_purpose=corporate_purpose,
        journey_type=journey_type,
    )

    try:
        result = await llm.complete(
            prompt=user_prompt,
            system_prompt="",
            cache_key=cache_key,
            call_type=call_type,
            cached_system_preamble=_CANON_PREAMBLE + "\n\n" + _CANON_SCHEMA,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("canon_llm_call_failed", extra={"error": str(e)})
        return None

    if not result:
        return None

    return _validate_response(result, agent_stage_names={s["stage_name"] for s in deduped})


def _build_user_prompt(
    *,
    mode: CanonMode,
    deduped: list[dict],
    industry_template: list[dict],
    industry: str,
    corporate_purpose: str,
    journey_type: JourneyType,
) -> str:
    template_lines = "\n".join(
        f"  - {s.get('name', '')}: {s.get('description', '')}"
        for s in industry_template if s.get("name")
    ) or "  (no template available)"

    def _fmt_stage(s: dict) -> str:
        journeys = ", ".join(s["source_journeys"]) or "?"
        desc = s["description"] or "(no description)"
        goal = f" (goal: {s['customer_goal']})" if s["customer_goal"] else ""
        return f"  [{journeys}] {s['stage_name']} — {desc}{goal}"

    agent_lines = "\n".join(_fmt_stage(s) for s in deduped) or "  (no agent stages)"

    return f"""Journey type: {journey_type}
Industry: {industry or 'unknown'}
Corporate purpose: {corporate_purpose or 'N/A'}
Mode: {mode}

Industry reference template:
{template_lines}

Agent-derived stages ({len(deduped)} unique):
{agent_lines}

Produce the canonical chronological stage list. Respond with JSON only."""


def _validate_response(data: dict, agent_stage_names: set[str]) -> dict | None:
    """Drop malformed entries; require ≥ 3 valid stages else return None."""
    if not isinstance(data, dict):
        return None
    raw = data.get("stages")
    if not isinstance(raw, list) or not raw:
        return None

    journey_name = str(data.get("journey_name") or "Journey").strip() or "Journey"
    used_ids: set[str] = set()
    out: list[CanonStage] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").strip()
        if not name:
            continue
        cid = _slugify(name)
        # Disambiguate id collisions
        base = cid
        i = 2
        while cid in used_ids:
            cid = f"{base}-{i}"
            i += 1
        used_ids.add(cid)
        synonyms = s.get("synonyms") or []
        synonyms = [str(x).strip() for x in synonyms if str(x).strip()] if isinstance(synonyms, list) else []
        source_journeys = s.get("source_journeys") or []
        source_journeys = [str(x).strip() for x in source_journeys if str(x).strip()] if isinstance(source_journeys, list) else []
        match = s.get("industry_template_match")
        if match is not None:
            match = str(match).strip() or None
        out.append(CanonStage(
            canon_id=cid,
            name=name,
            description=str(s.get("description") or "").strip(),
            customer_goal=str(s.get("customer_goal") or "").strip(),
            synonyms=synonyms[:8],
            source_journeys=source_journeys,
            source_stage_names=[name] if name in agent_stage_names else [],
            industry_template_match=match,
        ))

    if len(out) < 3:
        return None
    return {"journey_name": journey_name, "stages": out}


# ---------- Helpers ----------


def _stages_from_deduped(deduped: list[dict]) -> list[CanonStage]:
    """Build CanonStage list directly from dedup output (no LLM)."""
    used_ids: set[str] = set()
    out: list[CanonStage] = []
    for rec in deduped:
        name = rec["stage_name"]
        cid = _slugify(name)
        base = cid
        i = 2
        while cid in used_ids:
            cid = f"{base}-{i}"
            i += 1
        used_ids.add(cid)
        out.append(CanonStage(
            canon_id=cid, name=name,
            description=rec["description"],
            customer_goal=rec["customer_goal"],
            synonyms=[], source_journeys=rec["source_journeys"],
            source_stage_names=[name],
        ))
    return out


def _stages_from_template(template: list[dict]) -> list[CanonStage]:
    used_ids: set[str] = set()
    out: list[CanonStage] = []
    for s in template:
        name = (s.get("name") or "").strip() if isinstance(s, dict) else ""
        if not name:
            continue
        cid = _slugify(name)
        base = cid
        i = 2
        while cid in used_ids:
            cid = f"{base}-{i}"
            i += 1
        used_ids.add(cid)
        out.append(CanonStage(
            canon_id=cid, name=name,
            description=str(s.get("description") or "").strip(),
            customer_goal="", synonyms=[],
            source_journeys=[], source_stage_names=[],
            industry_template_match=name,
        ))
    return out


def _slugify(text: str) -> str:
    out: list[str] = []
    prev_dash = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash and out:
            out.append("-")
            prev_dash = True
    return "".join(out).rstrip("-") or "stage"


def _default_journey_name(journey_type: JourneyType, industry: str) -> str:
    if journey_type == "CX":
        return f"{industry} Customer Journey".strip() if industry else "Customer Journey"
    return f"{industry} Employee Journey".strip() if industry else "Employee Journey"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_hash(
    deduped: list[dict],
    industry_template: list[dict],
    journey_type: JourneyType,
    mode: str,
) -> str:
    payload = {
        "v": CANON_PROMPT_VERSION,
        "journey_type": journey_type,
        "mode": mode,
        "agent": [{"n": s["stage_name"], "j": s["source_journeys"]} for s in deduped],
        "template": [{"n": s.get("name", "")} for s in industry_template if isinstance(s, dict)],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
