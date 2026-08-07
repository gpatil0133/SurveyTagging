"""Top-level required-key validators for tenant_profile artifacts.

We deliberately do NOT enforce strict per-field JSON Schema validation. The
agent prompts ask for ~30+ deeply nested fields and the LLM returns "Unknown"
or null for missing data. A strict schema would reject perfectly usable
artifacts. Instead we check that the top-level sections exist (so consumers
can rely on their presence) and report missing/unexpected keys as warnings
that get persisted alongside the artifact.

Per-field downstream consumers should treat all leaf fields as optional and
fall back gracefully on missing keys.
"""

from __future__ import annotations

# Top-level keys promised by each agent prompt's OUTPUT SCHEMA.
ORG_REQUIRED_KEYS: frozenset[str] = frozenset({
    "organization_profile",
    "classification",
    "operational_scope",
    "technology_ecosystem",
    "digital_presence",
    "market_position",
    "confidence",
    "metadata",
})

CX_REQUIRED_KEYS: frozenset[str] = frozenset({
    "customer_profile",
    "customer_types",
    "journeys",
    "confidence",
    "metadata",
})

EX_REQUIRED_KEYS: frozenset[str] = frozenset({
    "workforce_profile",
    "employee_types",
    "lifecycle_analysis",
    "ex_maturity",
    "confidence",
    "metadata",
})


def validate(payload: dict, agent: str) -> list[str]:
    """Return a list of human-readable validation warnings (empty == clean).

    Currently checks: payload is a dict, top-level required keys present,
    `confidence.overall` is a non-empty string when present.
    """
    warnings: list[str] = []
    if not isinstance(payload, dict):
        warnings.append(f"payload is not a dict (got {type(payload).__name__})")
        return warnings

    required = {"org": ORG_REQUIRED_KEYS, "cx": CX_REQUIRED_KEYS, "ex": EX_REQUIRED_KEYS}.get(agent)
    if required is None:
        warnings.append(f"unknown agent {agent!r} — no schema to check against")
        return warnings

    missing = sorted(required - payload.keys())
    if missing:
        warnings.append(f"missing top-level keys: {missing}")

    confidence = payload.get("confidence")
    if isinstance(confidence, dict) and not confidence.get("overall"):
        warnings.append("confidence.overall is empty or missing")

    return warnings
