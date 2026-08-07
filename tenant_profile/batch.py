"""Batch driver — runs the org/CX/EX fetcher across many tenants from YAML.

Sequential by design: `pro` processor takes ~10 min per call, so 5 tenants ×
3 agents ≈ 2.5 hr. Parallelizing buys speed at the cost of API quota burn
and noisier logs; defer until we have a clear need.

Reads `survey_tagging/config/tenant_websites.yaml` (or any path passed via
`--inputs`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tenant_profile.parallel_client import ParallelClient, ParallelClientError
from tenant_profile.runner import (
    Agent,
    FetchResult,
    run_org,
    run_cx,
    run_ex,
)

logger = logging.getLogger(__name__)

DEFAULT_AGENTS: tuple[Agent, ...] = ("org", "cx", "ex")
_AGENT_RUNNERS = {"org": run_org, "cx": run_cx, "ex": run_ex}


@dataclass
class TenantSpec:
    tenant_id: int
    website_url: str
    agents: tuple[Agent, ...] = DEFAULT_AGENTS
    notes: str = ""


@dataclass
class BatchResult:
    successes: list[FetchResult] = field(default_factory=list)
    cache_hits: list[FetchResult] = field(default_factory=list)
    failures: list[tuple[int, Agent, str]] = field(default_factory=list)  # (tenant_id, agent, error)


def load_tenant_specs(inputs_path: Path) -> list[TenantSpec]:
    """Parse + validate config/tenant_websites.yaml."""
    if not inputs_path.exists():
        raise FileNotFoundError(f"tenant_websites file not found: {inputs_path}")
    raw = yaml.safe_load(inputs_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("tenants") or []
    specs: list[TenantSpec] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"tenants[{i}] is not a mapping: {entry!r}")
        try:
            tenant_id = int(entry["tenant_id"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"tenants[{i}] has missing/invalid tenant_id: {e}") from e
        website_url = str(entry.get("website_url") or "").strip()
        if not website_url:
            raise ValueError(f"tenants[{i}] (tenant_id={tenant_id}) has empty website_url")
        agents_raw = entry.get("agents") or list(DEFAULT_AGENTS)
        agents: list[Agent] = []
        for a in agents_raw:
            if a not in _AGENT_RUNNERS:
                raise ValueError(
                    f"tenants[{i}] (tenant_id={tenant_id}) has unknown agent {a!r}; "
                    f"allowed: {sorted(_AGENT_RUNNERS)}"
                )
            agents.append(a)  # type: ignore[arg-type]
        specs.append(TenantSpec(
            tenant_id=tenant_id, website_url=website_url,
            agents=tuple(agents), notes=str(entry.get("notes") or ""),
        ))
    return specs


def run_batch(
    specs: list[TenantSpec],
    output_dir: Path,
    client: ParallelClient,
    force: bool = False,
    only: tuple[Agent, ...] | None = None,
    skip: tuple[Agent, ...] = (),
) -> BatchResult:
    """Iterate specs sequentially, running each tenant's agents in dependency order.

    Args:
        only: If set, run only these agents (overrides spec.agents). E.g. ("org",).
        skip: Agents to skip across the whole batch.
    """
    result = BatchResult()
    for spec in specs:
        agents_to_run = list(only or spec.agents)
        agents_to_run = [a for a in agents_to_run if a not in skip]
        # Always run org first when present — CX/EX depend on its artifact.
        agents_to_run.sort(key=lambda a: 0 if a == "org" else 1)

        for agent in agents_to_run:
            runner = _AGENT_RUNNERS[agent]
            try:
                fr = runner(
                    tenant_id=spec.tenant_id,
                    website_url=spec.website_url,
                    output_dir=output_dir,
                    client=client,
                    force=force,
                )
                if fr.cached:
                    result.cache_hits.append(fr)
                else:
                    result.successes.append(fr)
            except ParallelClientError as e:
                logger.error("tenant_agent_failed",
                             extra={"tenant_id": spec.tenant_id, "agent": agent, "error": str(e)})
                result.failures.append((spec.tenant_id, agent, str(e)))
                # If org fails, skip CX/EX for this tenant — they require it.
                if agent == "org":
                    for downstream in ("cx", "ex"):
                        if downstream in agents_to_run:
                            result.failures.append((
                                spec.tenant_id, downstream,  # type: ignore[arg-type]
                                "skipped: org_profile fetch failed",
                            ))
                    break
    return result


def render_summary(result: BatchResult) -> str:
    """Human-readable batch summary for CLI output."""
    lines = [
        f"Fetched: {len(result.successes)}",
        f"Cache hits: {len(result.cache_hits)}",
        f"Failures: {len(result.failures)}",
    ]
    if result.failures:
        lines.append("")
        lines.append("Failures:")
        for tid, agent, err in result.failures:
            lines.append(f"  - tenant {tid} / {agent}: {err}")
    return "\n".join(lines)
