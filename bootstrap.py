"""Composition root — the single place that wires the system together.

Replaces the previously-duplicated construction blocks (api._init_globals,
api._build_orchestrator_with_llm, main.py's per-command LLMClient/orchestrator
wiring). Build one `AppContext` at process startup and pass it everywhere.

    ctx = build_context()                 # loads taxonomy, taggers, llm
    orch = build_orchestrator(ctx)        # tenant-level coordinator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import sharefs
from config_loaders.industry_stages import IndustryStagesRegistry
from models.taxonomy import TaxonomyRegistry
from settings import Settings
from taggers.registry import TaggerRegistry

logger = logging.getLogger("survey_tagging.bootstrap")

CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class AppContext:
    """Everything the tagging surface needs, built once and shared read-only.

    `llm_client` is None when LLM is disabled (skip_llm) or its init failed —
    callers treat that as deterministic-only. The LLM client is safe to reuse
    across surveys: each survey runs its two LLM calls in its own event loop.
    """

    settings: Settings
    taxonomy: TaxonomyRegistry
    registry: TaggerRegistry
    industry_stages: IndustryStagesRegistry
    llm_client: object | None  # llm.client.LLMClient | None (lazy import)
    config_dir: Path = CONFIG_DIR

    @property
    def llm_enabled(self) -> bool:
        return self.llm_client is not None


def build_llm_client(settings: Settings, *, skip_llm: bool | None = None):
    """Build the LLMClient from settings, or return None.

    Returns None when skip_llm is in effect or construction fails (missing key,
    import error) — the pipeline then falls back to deterministic-only tagging.
    `skip_llm` overrides `settings.skip_llm` when passed explicitly.
    """
    skip = settings.skip_llm if skip_llm is None else skip_llm
    if skip:
        logger.info("llm_disabled (skip_llm)")
        return None
    try:
        from llm.client import LLMClient
        client = LLMClient(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            rate_limit_rpm=settings.llm_rate_limit_rpm,
            cache_dir=settings.cache_dir,
            use_prompt_caching=settings.llm_use_prompt_caching,
        )
        logger.info("llm_client_configured",
                    extra={"model": settings.llm_model,
                           "prompt_caching": settings.llm_use_prompt_caching})
        return client
    except Exception as e:  # noqa: BLE001
        logger.warning("llm_client_init_failed", extra={"error": str(e)})
        return None


def build_context(settings: Settings | None = None, *, skip_llm: bool | None = None) -> AppContext:
    """Load taxonomy + taggers + industry stages + LLM client once."""
    settings = settings or Settings()

    # Record share credentials before anything can touch the share. This is
    # pure local state — it opens no socket, so it cannot fail or hang.
    #
    # The SMB session itself is established lazily, on the first share read
    # (sharefs.connect is memoized per server). Startup deliberately does NO
    # network I/O: the image server is often unreachable from a dev machine or
    # during a deploy window, and a 21-second TCP timeout there used to take the
    # whole process down before a single route was registered — including
    # /docs, the static UI and /api/health/share, the one endpoint whose job is
    # to report that the share is down.
    #
    # What the eager connect bought was diagnosis: a wrong password otherwise
    # surfaces as an empty tenant list, because every discovery function
    # swallows OSError and cannot tell "no surveys" from "cannot log in". That
    # is now `sharefs.probe()` behind GET /api/health/share, which reports the
    # underlying error instead of hiding it. `check_share.py` still probes
    # eagerly for the pre-flight check.
    sharefs.configure(settings.image_user, settings.image_pass)
    if sharefs.is_unc(settings.share_root or ""):
        logger.info("share_root_is_unc_lazy_connect",
                    extra={"root": str(settings.share_root)})

    taxonomy = TaxonomyRegistry.from_yaml(CONFIG_DIR / "taxonomy.yaml")
    logger.info("taxonomy_loaded", extra={"dimensions": len(taxonomy.all_dimensions)})

    industry_stages = IndustryStagesRegistry.from_yaml(CONFIG_DIR / "journey_stages.yaml")

    registry = TaggerRegistry()
    registry.discover("taggers.project")
    registry.discover("taggers.question")
    logger.info("taggers_registered", extra={"count": len(registry.all_taggers)})

    llm_client = build_llm_client(settings, skip_llm=skip_llm)

    return AppContext(
        settings=settings,
        taxonomy=taxonomy,
        registry=registry,
        industry_stages=industry_stages,
        llm_client=llm_client,
    )


def build_orchestrator(ctx: AppContext):
    """Construct the tenant-level PipelineOrchestrator from a context."""
    from pipeline.orchestrator import PipelineOrchestrator
    return PipelineOrchestrator(
        settings=ctx.settings,
        registry=ctx.registry,
        taxonomy=ctx.taxonomy,
        llm_client=ctx.llm_client,
        industry_stages=ctx.industry_stages,
    )
