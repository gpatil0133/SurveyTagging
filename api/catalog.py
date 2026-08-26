"""Catalog routes — process-wide, read-only, no tenant in the path.

Lifted out of run.py unchanged — see api/__init__.py for why the split."""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Header, HTTPException

import auth
import discovery
from api import deps

logger = logging.getLogger("survey_tagging.api")

router = APIRouter()


# ====================================================================
# §5  Catalog — process-wide, read-only, no tenant in the path
#     Ordered by cost, cheapest first: three in-memory reads (taxonomy,
#     config, me), then one that touches the share (health), then the one
#     that walks the entire share (surveys) and is kept for legacy callers.
# ====================================================================

@router.get("/api/taxonomy")
async def get_taxonomy() -> dict:
    """Full taxonomy — client dropdowns, plus the explanation layer the UI's
    Taxonomy tab renders (`explanation` / `derivation` / `strategy` /
    `purpose` / `feeds`).

    `purpose` is the one-line "why does this exist"; `feeds` names which outcome
    consumes it (S1..S6 for the planned experience-platform services, plus the
    live `Pipeline` / `Reporting` / `None` tokens). See config/taxonomy.yaml's
    header for the token legend.

    Covers all three levels; tenant dims are in here too, so a caller reading a
    tenant_tags.json artifact can look its dimensions up in the same catalog.
    """
    dims = {}
    for name, dim in deps.ctx.taxonomy.all_dimensions.items():
        dims[name] = {
            "level": dim.level,
            "description": dim.description,
            "purpose": dim.purpose,
            "feeds": dim.feeds,
            "explanation": dim.explanation,
            "derivation": dim.derivation,
            "strategy": dim.strategy,
            "allowed_values": dim.allowed_values,
            "multi_label": dim.multi_label,
            "user_defined": dim.user_defined,
            "canonical_values": dim.canonical_values,
            # Usually empty. Present on crosstab_axis_role, where it publishes
            # the filter/segment/search mapping that `control_role` used to
            # state as a dimension of its own (removed in V7.3).
            "derived_controls": dim.derived_controls,
        }
    return dims


@functools.lru_cache(maxsize=1)
def _widget_map() -> dict:
    """config/widget_map.yaml, read once per process.

    Cached because it is static config served on a read path; restart to pick up an
    edit, same as taxonomy.yaml.
    """
    path = Path(__file__).resolve().parent.parent / "config" / "widget_map.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@router.get("/api/widget-map")
async def widget_map() -> dict:
    """Our widget labels translated into the consumer's typed enums.

    `widget_compatibility` and `visualization_type` are prose labels; the dashboard
    consumer's contract is `WidgetType x ChartType x MetricType`. This is the bridge,
    so each consumer stops writing its own copy and drifting from the others.

    Every entry carries `confirmed`. False means the platform tokens have not been
    checked against the frontend's enum files (which live in the consumer repo) —
    the label and shape are right, the tokens are the gap. A row with no
    `chart_type` is usually a finding rather than an omission: "Trend Line" is a
    display mode on the rating widget, and "Heat Map" and "Ranking Bar" have no
    chart type at all. Read `widget_type` first; treat `chart_type` as optional.

    `reads_alongside` names the dimensions a consumer needs to finish a panel
    definition (what to aggregate, what to call it, what to slice by) — none of
    which belongs in this map.
    """
    try:
        data = _widget_map()
    except (OSError, yaml.YAMLError) as e:
        raise HTTPException(500, f"widget_map.yaml could not be read: {e}") from e

    widgets = data.get("widgets") or {}
    return {
        "version": data.get("version"),
        "widgets": widgets,
        "reads_alongside": data.get("reads_alongside") or {},
        # So a caller can tell at a glance how much of the map is verified without
        # walking every entry.
        "counts": {
            "total": len(widgets),
            "confirmed": sum(1 for w in widgets.values() if w.get("confirmed")),
            "unconfirmed": sum(1 for w in widgets.values() if not w.get("confirmed")),
        },
    }


@router.get("/api/config")
async def ui_config() -> dict:
    """Server config the UI needs to shape itself.

    `profile_source` decides the whole tenant-profile panel: the Parallel path
    needs a website and blocks for 10-30 minutes, the SMX path needs neither and
    can trigger generation on a miss. The browser has no other way to know which
    is configured.
    """
    return {
        # The sub-path this app is mounted under, "" at the origin root. The UI
        # already got it baked into index.html, so this is for non-browser
        # callers and for confirming what the server thinks it is when a
        # deployment's links come out wrong.
        "path_prefix": deps.settings.path_prefix,
        "profile_source": deps.settings.profile_source,
        "smx_allow_generate": deps.settings.smx_allow_generate,
        # Whether apismx can be called with NO caller token — i.e. whether the
        # server holds a headless fallback. The browser knows if it has a token
        # of its own; only this tells it whether the absence is fatal, which is
        # what lets the profile panel refuse up front instead of after a round
        # trip. The token itself is never exposed.
        "smx_token_configured": bool((deps.settings.smx_token or "").strip()),
        "smx_generate_wait_seconds": int(
            deps.settings.smx_generate_poll_attempts * deps.settings.smx_generate_poll_interval
        ),
        # So "the trace is on" is checkable without reading the server's .env.
        "smx_debug_wire": deps.settings.smx_debug_wire,
        "skip_llm": deps.settings.skip_llm,
    }


@router.get("/api/me")
async def whoami(authorization: str | None = Header(default=None)) -> dict:
    """What the caller's Bearer token says about them.

    `{corp_no, subject, has_token, verified, auth_enabled}`. The UI calls this
    once at boot: when it is embedded in the platform shell there is a token in
    localStorage but no corp number typed into the box, and this is how it
    learns which tenant to open. A typed corp number always overrides it.

    `verified` is False when the signature could not be checked (no/again wrong
    public key) but the claims were read anyway — see auth._decode_verified. It
    is advisory while `auth_enabled` is False.
    """
    return {**auth.principal(authorization), "auth_enabled": deps.settings.auth_enabled}


@router.get("/api/health/share")
async def share_health() -> dict:
    """Is the data root reachable? Lets the UI distinguish a downed share from
    a tenant that simply has no surveys (both otherwise look like an empty list)."""
    return await asyncio.to_thread(discovery.probe_root, deps.settings.data_dir)


@router.get("/api/surveys")
async def list_surveys() -> list[dict]:
    """List all tenants and their surveys discovered under the local data dir.

    Walks the whole data root — expensive over a network share (minutes, not
    seconds). The current UI does not call it; prefer
    GET /api/tenants/{t}/tag-surveys, which is one directory listing.

    Offloaded to a thread even though it is the only caller's own problem how
    long it takes: run inline in the event loop, one request to this route
    freezes the entire server for everyone — no other route answers, not even
    /api/health/share, and the process looks hung rather than busy. A stale
    browser tab still holding the pre-v7 UI calls this on load, which is exactly
    how that happens in practice.
    """
    return await asyncio.to_thread(discovery.discover_catalog, deps.settings.data_dir)
