"""Ops CLI for the survey tagging system.

Tagging itself is **API-only** (see run.py / the FastAPI app) — there is no
`tag` subcommand. This CLI exists only for the Parallel.ai tenant-profile
fetcher, which is an onboarding/ops task run out-of-band from tagging.

  survey-tagger profile fetch     --tenant ... --website ...
  survey-tagger profile fetch-all --inputs config/tenant_websites.yaml
  survey-tagger profile show      --tenant ... --type org
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

# Add current directory to path for relative imports when running as CLI
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Load .env into os.environ so libraries that read env vars directly
# (the parallel-web SDK reads PARALLEL_API_KEY) see the same values Pydantic
# Settings does. Pydantic Settings reads .env into its own attributes but does
# NOT export to os.environ, so without this hop bare keys in .env are invisible
# to those libraries.
try:
    from dotenv import load_dotenv
    load_dotenv(_current_dir / ".env", override=False)
except ImportError:
    pass

import tls_trust
from settings import Settings


def setup_logging(level: str, fmt: str) -> None:
    from log_config import configure_logging
    configure_logging(level)


@click.group()
def cli() -> None:
    """Survey Auto-Tagging — ops CLI (tenant-profile fetcher only; tagging is API-only)."""


@cli.group()
def profile() -> None:
    """Tenant profile fetcher: outsources website-driven research to Parallel.ai."""


_AGENT_CHOICES = click.Choice(["org", "cx", "ex"], case_sensitive=False)


def _build_parallel_client(settings: Settings) -> "ParallelClient":  # noqa: F821
    from tenant_profile.parallel_client import ParallelClient, ParallelClientError
    if not settings.parallel_api_key:
        raise click.ClickException(
            "PARALLEL_API_KEY not set. Add SURVEY_TAGGER_PARALLEL_API_KEY=... to .env."
        )
    try:
        return ParallelClient(
            api_key=settings.parallel_api_key,
            processor=settings.parallel_processor,
            api_timeout=settings.parallel_api_timeout,
            max_retries=settings.parallel_max_retries,
        )
    except ParallelClientError as e:
        raise click.ClickException(str(e))


@profile.command("fetch")
@click.option("--tenant", type=int, required=True, help="Tenant ID")
@click.option("--website", type=str, required=True, help="Website URL for this tenant")
@click.option("--only", type=_AGENT_CHOICES, multiple=True,
              help="Run only these agents (default: org,cx,ex)")
@click.option("--skip", type=_AGENT_CHOICES, multiple=True,
              help="Skip these agents")
@click.option("--force", is_flag=True, default=False,
              help="Re-fetch even if artifact exists on disk")
@click.option("--output-dir", type=click.Path(), default=None)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default=None)
def profile_fetch(
    tenant: int, website: str,
    only: tuple[str, ...], skip: tuple[str, ...], force: bool,
    output_dir: str | None, log_level: str | None,
) -> None:
    """Fetch tenant profile (org -> cx -> ex) for a single tenant."""
    settings = Settings()
    if output_dir:
        settings.output_dir = Path(output_dir)
    if log_level:
        settings.log_level = log_level
    setup_logging(settings.log_level, settings.log_format)
    # The CLI builds Settings directly instead of going through
    # bootstrap.build_context, so it has to install the OS trust store itself
    # or it hits CERTIFICATE_VERIFY_FAILED wherever the service used to.
    tls_trust.install(settings)

    from tenant_profile.batch import (
        TenantSpec, DEFAULT_AGENTS, run_batch, render_summary,
    )

    only_norm = tuple(a.lower() for a in only) or None
    skip_norm = tuple(a.lower() for a in skip)
    spec = TenantSpec(tenant_id=tenant, website_url=website, agents=DEFAULT_AGENTS)
    client = _build_parallel_client(settings)
    result = run_batch(
        specs=[spec], output_dir=Path(settings.output_dir),
        client=client, force=force,
        only=only_norm, skip=skip_norm,
    )
    click.echo(render_summary(result))
    if result.failures:
        sys.exit(1)


@profile.command("fetch-all")
@click.option("--inputs", type=click.Path(exists=True), default=None,
              help="Path to tenant_websites.yaml (default: config/tenant_websites.yaml)")
@click.option("--only", type=_AGENT_CHOICES, multiple=True,
              help="Run only these agents")
@click.option("--skip", type=_AGENT_CHOICES, multiple=True,
              help="Skip these agents")
@click.option("--force", is_flag=True, default=False)
@click.option("--output-dir", type=click.Path(), default=None)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default=None)
def profile_fetch_all(
    inputs: str | None,
    only: tuple[str, ...], skip: tuple[str, ...], force: bool,
    output_dir: str | None, log_level: str | None,
) -> None:
    """Batch-fetch tenant profiles from config/tenant_websites.yaml."""
    settings = Settings()
    if output_dir:
        settings.output_dir = Path(output_dir)
    if log_level:
        settings.log_level = log_level
    setup_logging(settings.log_level, settings.log_format)
    # The CLI builds Settings directly instead of going through
    # bootstrap.build_context, so it has to install the OS trust store itself
    # or it hits CERTIFICATE_VERIFY_FAILED wherever the service used to.
    tls_trust.install(settings)

    inputs_path = Path(inputs) if inputs else (
        Path(__file__).parent / "config" / "tenant_websites.yaml"
    )
    from tenant_profile.batch import load_tenant_specs, run_batch, render_summary
    specs = load_tenant_specs(inputs_path)
    if not specs:
        click.echo(f"No tenants in {inputs_path} — nothing to do.", err=True)
        return
    click.echo(f"Loaded {len(specs)} tenant(s) from {inputs_path}")

    only_norm = tuple(a.lower() for a in only) or None
    skip_norm = tuple(a.lower() for a in skip)
    client = _build_parallel_client(settings)
    result = run_batch(
        specs=specs, output_dir=Path(settings.output_dir),
        client=client, force=force,
        only=only_norm, skip=skip_norm,
    )
    click.echo(render_summary(result))
    if result.failures:
        sys.exit(1)


@profile.command("show")
@click.option("--tenant", type=int, required=True)
@click.option("--type", "agent_type", type=_AGENT_CHOICES, default="org",
              help="Which artifact to show (default: org)")
@click.option("--raw/--envelope", default=False,
              help="--raw prints just agent_output; default prints the full envelope")
def profile_show(tenant: int, agent_type: str, raw: bool) -> None:
    """Print a tenant profile artifact to stdout."""
    settings = Settings()
    from tenant_profile.runner import artifact_path, load_artifact
    path = artifact_path(tenant, agent_type.lower(), settings.profile_root)  # type: ignore[arg-type]
    envelope = load_artifact(path)
    if envelope is None:
        click.echo(f"No artifact at {path}", err=True)
        sys.exit(1)
    payload = envelope.get("agent_output") if raw else envelope
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
