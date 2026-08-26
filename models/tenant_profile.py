"""TenantProfile model — read-only loader for the three Parallel.ai artifacts.

Wraps `output/{tenant_id}/tenant_profile/{org,cx,ex}_*.json` and exposes the
specific fields that downstream consumers (Phase 3 stage short-circuit,
Phase 4 tagger priors, tenant-level org context) need.

Design notes:
- Raw artifact payloads (`org_payload`, `cx_payload`, `ex_payload`) are kept
  as dicts. Accessor properties read into them with safe defaults — agents
  return sparse / partial structures, so every leaf is treated as optional.
- The loader is fault-tolerant: missing or unreadable artifacts produce
  `None` for that part rather than raising. Callers should check `has_org`
  etc. or use the convenience accessors which return safe defaults.
- This module is purely a reader. Writing artifacts is the
  `tenant_profile.runner` module's job.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TenantProfile(BaseModel):
    """Aggregated read-side view of org / CX / EX artifacts for one tenant."""

    tenant_id: int

    # Raw envelope payloads from each agent (the `agent_output` dict, not the
    # full envelope). Any of the three may be None if not yet fetched or if
    # the artifact failed to parse.
    org_payload: dict[str, Any] | None = None
    cx_payload: dict[str, Any] | None = None
    ex_payload: dict[str, Any] | None = None

    # Source paths — handy for debugging / logging which artifacts populated
    # this profile. Empty list if loaded from in-memory dicts.
    artifact_paths: list[str] = Field(default_factory=list)

    # ---------- Loader ----------

    @classmethod
    def load(cls, tenant_id: int, output_dir: Path) -> "TenantProfile | None":
        """Read all three artifacts from disk. Returns None if none exist.

        A profile with only org_profile is valid (CX/EX may be deferred or
        skipped per tenant_websites.yaml). A profile with no artifacts at all
        returns None — caller should treat that as "not fetched yet" and
        either trigger a fetch or fall back to directory / survey signals.
        """
        from tenant_profile.runner import artifact_path, load_artifact

        org_path = artifact_path(tenant_id, "org", output_dir)
        cx_path = artifact_path(tenant_id, "cx", output_dir)
        ex_path = artifact_path(tenant_id, "ex", output_dir)

        org_env = load_artifact(org_path)
        cx_env = load_artifact(cx_path)
        ex_env = load_artifact(ex_path)

        if org_env is None and cx_env is None and ex_env is None:
            return None

        paths: list[str] = []
        for env, p in ((org_env, org_path), (cx_env, cx_path), (ex_env, ex_path)):
            if env is not None:
                paths.append(str(p))

        return cls(
            tenant_id=tenant_id,
            org_payload=_extract_agent_output(org_env),
            cx_payload=_extract_agent_output(cx_env),
            ex_payload=_extract_agent_output(ex_env),
            artifact_paths=paths,
        )

    # ---------- Presence flags ----------

    @property
    def has_org(self) -> bool:
        return isinstance(self.org_payload, dict) and bool(self.org_payload)

    @property
    def has_cx(self) -> bool:
        return isinstance(self.cx_payload, dict) and bool(self.cx_payload)

    @property
    def has_ex(self) -> bool:
        return isinstance(self.ex_payload, dict) and bool(self.ex_payload)

    @property
    def is_empty(self) -> bool:
        return not (self.has_org or self.has_cx or self.has_ex)

    # ---------- Org-level accessors (Phase 2 + 4) ----------

    @property
    def corporate_name(self) -> str:
        """Tenant's display name."""
        return _get_str(self.org_payload, ["organization_profile", "name"])

    @property
    def corporate_purpose(self) -> str:
        """Tenant's mission / value proposition.

        Prefers `mission`, falls back to `value_proposition`, then `description`.
        """
        for key in ("mission", "value_proposition", "description"):
            v = _get_str(self.org_payload, ["organization_profile", key])
            if v:
                return v
        return ""

    @property
    def industry_vertical(self) -> str:
        """Primary industry. Phase 4 prior for `industry_vertical` tag."""
        return _get_str(self.org_payload, ["classification", "industry", "primary"])

    @property
    def industry_sub_vertical(self) -> str:
        return _get_str(self.org_payload, ["classification", "industry", "sub_vertical"])

    @property
    def industry_taxonomy_vertical(self) -> str:
        """Agent's industry coerced into the canonical taxonomy/registry key.

        The agent emits free-form labels ("Healthcare & Life Sciences", "SaaS /
        Cloud Software") while taxonomy.yaml uses a fixed short name
        ("Healthcare", "SaaS / Technology"). This accessor maps between them so
        the deterministic industry fallback matches.
        """
        return _normalize_agent_industry(self.industry_vertical)

    @property
    def regulatory_intensity(self) -> str:
        return _get_str(self.org_payload, ["operational_scope", "regulatory", "intensity"])

    @property
    def regulatory_frameworks(self) -> list[str]:
        v = _get_path(self.org_payload, ["operational_scope", "regulatory", "frameworks"])
        return [str(x) for x in v] if isinstance(v, list) else []

    @property
    def data_sensitivity(self) -> str:
        """Phase 4 prior for `data_sensitivity` question tag."""
        return _get_str(self.org_payload, ["operational_scope", "regulatory", "data_sensitivity"])

    @property
    def geographic_reach(self) -> str:
        return _get_str(self.org_payload, ["operational_scope", "geographic", "reach"])

    @property
    def org_confidence(self) -> str:
        return _get_str(self.org_payload, ["confidence", "overall"])

    # ---------- CX accessors (Phase 3 + 4) ----------

    @property
    def primary_customer_segment(self) -> str:
        """Phase 4 prior for `audience_type` (B2B/B2C → maps to audience)."""
        return _get_str(self.cx_payload, ["customer_profile", "primary_segment"])

    @property
    def secondary_customer_segments(self) -> list[str]:
        v = _get_path(self.cx_payload, ["customer_profile", "secondary_segments"])
        return [str(x) for x in v] if isinstance(v, list) else []

    @property
    def relationship_type(self) -> str:
        """Phase 4 prior for `relationship_type` project tag (high-confidence only)."""
        return _get_str(self.cx_payload, ["customer_profile", "relationship_type"])

    @property
    def sales_cycle(self) -> str:
        return _get_str(self.cx_payload, ["customer_profile", "sales_cycle"])

    @property
    def customer_types(self) -> list[dict[str, Any]]:
        """Phase 4 prior for `survey_sub_type` (medium-confidence + non-empty)."""
        v = _get_path(self.cx_payload, ["customer_types"])
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    @property
    def cx_journeys(self) -> list[dict[str, Any]]:
        """Raw journeys[] list — Phase 3 stage short-circuit input."""
        v = _get_path(self.cx_payload, ["journeys"])
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    @property
    def cx_confidence(self) -> str:
        return _get_str(self.cx_payload, ["confidence", "overall"])

    @property
    def cx_journeys_confidence(self) -> str:
        return _get_str(self.cx_payload, ["confidence", "journeys_confidence"])

    # ---------- EX accessors (Phase 3 + 4) ----------

    @property
    def workforce_composition(self) -> str:
        return _get_str(self.ex_payload, ["workforce_profile", "composition"])

    @property
    def work_arrangement(self) -> str:
        return _get_str(self.ex_payload, ["workforce_profile", "work_arrangement"])

    @property
    def frontline_ratio(self) -> str:
        return _get_str(self.ex_payload, ["workforce_profile", "frontline_ratio"])

    @property
    def employee_types(self) -> list[dict[str, Any]]:
        v = _get_path(self.ex_payload, ["employee_types"])
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    @property
    def ex_lifecycle_stages(self) -> list[dict[str, Any]]:
        """Raw lifecycle_analysis.stages[] — Phase 3 EX stage short-circuit input."""
        v = _get_path(self.ex_payload, ["lifecycle_analysis", "stages"])
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    @property
    def ex_maturity_level(self) -> str:
        return _get_str(self.ex_payload, ["ex_maturity", "level"])

    @property
    def ex_confidence(self) -> str:
        return _get_str(self.ex_payload, ["confidence", "overall"])

    @property
    def ex_lifecycle_confidence(self) -> str:
        return _get_str(self.ex_payload, ["confidence", "breakdown", "lifecycle"])


# ---------- Helpers ----------


def _extract_agent_output(envelope: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pull `agent_output` out of an artifact envelope. Returns None if missing or not a dict."""
    if envelope is None:
        return None
    payload = envelope.get("agent_output")
    return payload if isinstance(payload, dict) else None


def _get_path(payload: dict[str, Any] | None, path: list[str]) -> Any:
    """Walk a nested-dict path. Returns None on any missing step."""
    if payload is None:
        return None
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _get_str(payload: dict[str, Any] | None, path: list[str]) -> str:
    """Walk a path and stringify; empty string for any missing/non-scalar value."""
    v = _get_path(payload, path)
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    return ""


# Keyword fragments → taxonomy industry. Used to map the Parallel.ai agent's
# free-form industry strings ("Healthcare & Life Sciences", "SaaS / Cloud
# Software") onto canonical taxonomy/registry keys. Order matters — first
# match wins. Mirrors taggers.project.industry._AGENT_INDUSTRY_KEYWORDS.
_AGENT_INDUSTRY_KEYWORDS: list[tuple[str, str]] = [
    ("health", "Healthcare"),
    ("medical", "Healthcare"),
    ("pharma", "Healthcare"),
    ("financ", "Financial Services"),
    ("bank", "Financial Services"),
    ("insurance", "Financial Services"),
    ("higher education", "Higher Education"),
    ("university", "Higher Education"),
    ("k-12", "K-12 Education"),
    ("retail", "Retail / E-commerce"),
    ("e-commerce", "Retail / E-commerce"),
    ("ecommerce", "Retail / E-commerce"),
    ("hospitality", "Hospitality / Travel"),
    ("travel", "Hospitality / Travel"),
    ("hotel", "Hospitality / Travel"),
    ("saas", "SaaS / Technology"),
    ("software", "SaaS / Technology"),
    ("technology", "SaaS / Technology"),
    ("government", "Government / Public Sector"),
    ("public sector", "Government / Public Sector"),
    ("fitness", "Fitness & Wellness"),
    ("wellness", "Fitness & Wellness"),
]


def _normalize_agent_industry(agent_value: str) -> str:
    """Coerce agent industry string to taxonomy key, or "" if unmappable."""
    if not agent_value:
        return ""
    lower = agent_value.lower()
    for needle, tag in _AGENT_INDUSTRY_KEYWORDS:
        if needle in lower:
            return tag
    return ""
