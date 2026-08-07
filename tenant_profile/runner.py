"""Per-tenant fetcher: run org / CX / EX agents and persist artifacts.

Layout on disk:
    {output_dir}/{tenant_id}/tenant_profile/
        org_profile.json
        cx_intelligence.json
        ex_intelligence.json

Each artifact is wrapped in a thin envelope:
    {
      "schema_version": "1.0",
      "agent": "org" | "cx" | "ex",
      "tenant_id": int,
      "website_url": str,
      "fetched_at": ISO 8601,
      "parallel": { "run_id": str, "processor": str, "submitted_at": str, "completed_at": str },
      "validation_warnings": [str, ...],
      "agent_output": { ... raw payload from Parallel.ai ... }
    }

Idempotent: if the artifact already exists and `force=False`, returns the
cached envelope without calling Parallel. Pass `force=True` to refresh.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import sharefs
from typing import Any, Literal

from fs_utils import write_json_atomic
from tenant_profile.parallel_client import (
    ParallelClient,
    ParallelClientError,
    render_prompt,
)
from tenant_profile.schemas import validate as validate_payload

logger = logging.getLogger(__name__)

Agent = Literal["org", "cx", "ex"]
ARTIFACT_SCHEMA_VERSION = "1.0"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPT_FILES: dict[Agent, str] = {
    "org": "org_profile.txt",
    "cx":  "cx_intelligence.txt",
    "ex":  "ex_intelligence.txt",
}
_ARTIFACT_FILES: dict[Agent, str] = {
    "org": "org_profile.json",
    "cx":  "cx_intelligence.json",
    "ex":  "ex_intelligence.json",
}


class ArtifactExists(Exception):
    """Raised internally when a fetch is short-circuited by cache; surfaced
    from CLI as an info-level message, not an error."""


@dataclass
class FetchResult:
    """What a single agent run produced."""
    agent: Agent
    tenant_id: int
    artifact_path: Path
    cached: bool                           # True when we returned the on-disk artifact unchanged
    envelope: dict[str, Any] = field(default_factory=dict)


def artifact_path(tenant_id: int, agent: Agent, output_dir: Path) -> Path:
    return Path(output_dir) / str(tenant_id) / "tenant_profile" / _ARTIFACT_FILES[agent]


def load_artifact(path: Path) -> dict[str, Any] | None:
    if not sharefs.exists(path):
        return None
    try:
        return json.loads(sharefs.read_text(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("artifact_unreadable", extra={"path": str(path), "error": str(e)})
        return None


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write to a temp file in the same dir, then rename — avoids torn writes."""
    write_json_atomic(path, payload)


def _load_prompt(agent: Agent) -> str:
    prompt_file = _PROMPTS_DIR / _PROMPT_FILES[agent]
    if not prompt_file.exists():
        raise FileNotFoundError(f"missing prompt template: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def _build_envelope(
    agent: Agent,
    tenant_id: int,
    website_url: str,
    payload: dict[str, Any],
    parallel_meta: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "agent": agent,
        "tenant_id": tenant_id,
        "website_url": website_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "parallel": parallel_meta,
        "validation_warnings": warnings,
        "agent_output": payload,
    }


def _run_agent(
    agent: Agent,
    tenant_id: int,
    website_url: str,
    output_dir: Path,
    client: ParallelClient,
    template_vars: dict[str, str],
    input_payload: dict[str, Any],
    force: bool,
) -> FetchResult:
    path = artifact_path(tenant_id, agent, output_dir)

    if not force:
        cached = load_artifact(path)
        if cached is not None:
            logger.info("artifact_cache_hit",
                        extra={"agent": agent, "tenant_id": tenant_id, "path": str(path)})
            return FetchResult(agent=agent, tenant_id=tenant_id, artifact_path=path,
                               cached=True, envelope=cached)

    template = _load_prompt(agent)
    prompt = render_prompt(template, template_vars)

    logger.info("parallel_run_starting",
                extra={"agent": agent, "tenant_id": tenant_id, "website": website_url})
    result = client.run_task(prompt=prompt, input_payload=input_payload)
    warnings = validate_payload(result.payload, agent)
    if warnings:
        logger.warning("artifact_validation_warnings",
                       extra={"agent": agent, "tenant_id": tenant_id, "warnings": warnings})

    envelope = _build_envelope(
        agent=agent, tenant_id=tenant_id, website_url=website_url,
        payload=result.payload,
        parallel_meta={
            "run_id": result.meta.run_id,
            "processor": result.meta.processor,
            "submitted_at": result.meta.submitted_at,
            "completed_at": result.meta.completed_at,
        },
        warnings=warnings,
    )
    _write_atomic(path, envelope)
    logger.info("artifact_written",
                extra={"agent": agent, "tenant_id": tenant_id, "path": str(path),
                       "warning_count": len(warnings)})
    return FetchResult(agent=agent, tenant_id=tenant_id, artifact_path=path,
                       cached=False, envelope=envelope)


def run_org(
    tenant_id: int,
    website_url: str,
    output_dir: Path,
    client: ParallelClient,
    force: bool = False,
) -> FetchResult:
    """Fetch organization profile (depends on website only)."""
    return _run_agent(
        agent="org", tenant_id=tenant_id, website_url=website_url,
        output_dir=output_dir, client=client,
        template_vars={"website_url": website_url},
        input_payload={"website_url": website_url},
        force=force,
    )


def _require_org(tenant_id: int, output_dir: Path) -> dict[str, Any]:
    """CX and EX both depend on org_profile existing; load and surface its agent_output."""
    org_path = artifact_path(tenant_id, "org", output_dir)
    cached = load_artifact(org_path)
    if cached is None:
        raise ParallelClientError(
            f"cannot run CX/EX agent for tenant {tenant_id}: "
            f"org_profile.json missing at {org_path}. Run `profile fetch --only org` first."
        )
    payload = cached.get("agent_output")
    if not isinstance(payload, dict):
        raise ParallelClientError(
            f"org_profile.json for tenant {tenant_id} is malformed (no agent_output dict)."
        )
    return payload


def run_cx(
    tenant_id: int,
    website_url: str,
    output_dir: Path,
    client: ParallelClient,
    force: bool = False,
) -> FetchResult:
    """Fetch CX intelligence (depends on website + org_profile artifact).

    The org_profile is passed via the Parallel.ai `input` field (not embedded
    in the prompt) — the prompt's 15,000-char limit on `output_schema` doesn't
    apply to `input`, and the agent reads the data from there.
    """
    org_payload = _require_org(tenant_id, output_dir)
    return _run_agent(
        agent="cx", tenant_id=tenant_id, website_url=website_url,
        output_dir=output_dir, client=client,
        template_vars={"website_url": website_url},
        input_payload={"website_url": website_url, "org_profile": org_payload},
        force=force,
    )


def run_ex(
    tenant_id: int,
    website_url: str,
    output_dir: Path,
    client: ParallelClient,
    force: bool = False,
) -> FetchResult:
    """Fetch EX intelligence (depends on website + org_profile artifact).

    See run_cx — same input-vs-template separation applies.
    """
    org_payload = _require_org(tenant_id, output_dir)
    return _run_agent(
        agent="ex", tenant_id=tenant_id, website_url=website_url,
        output_dir=output_dir, client=client,
        template_vars={"website_url": website_url},
        input_payload={"website_url": website_url, "org_profile": org_payload},
        force=force,
    )
