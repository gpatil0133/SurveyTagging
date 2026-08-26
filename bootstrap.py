"""Composition root — the single place that wires the system together.

Replaces the previously-duplicated construction blocks (api._init_globals,
api._build_orchestrator_with_llm, main.py's per-command LLMClient/orchestrator
wiring). Build one `AppContext` at process startup and pass it everywhere.

    ctx = build_context()                 # loads taxonomy, taggers, llm
    orch = build_orchestrator(ctx)        # tenant-level coordinator
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import auth
import sharefs
import tls_trust
from models.taxonomy import TaxonomyRegistry
from pipeline.change_detector import ChangeDetector
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
    # ONE per process. It owns `survey_hashes.json`, whose in-memory copy is
    # rewritten wholesale on every `mark_processed`: a second instance means two
    # copies of that dict, two locks, and whichever saves last silently drops the
    # other's entries — surveys then re-tag forever or keep a hash whose output
    # was never written. Build it here, never in a request path.
    change_detector: ChangeDetector
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
            num_retries=settings.llm_num_retries,
        )
        logger.info("llm_client_configured",
                    extra={"model": settings.llm_model,
                           "prompt_caching": settings.llm_use_prompt_caching})
        return client
    except Exception as e:  # noqa: BLE001
        logger.warning("llm_client_init_failed", extra={"error": str(e)})
        return None


def build_context(settings: Settings | None = None, *, skip_llm: bool | None = None) -> AppContext:
    """Load taxonomy + taggers + LLM client once."""
    settings = settings or Settings()
    auth.configure(settings)
    _check_deployment_safety(settings)

    # Before any outbound TLS. The SoGo certs chain to a corporate root that
    # lives in the OS trust store and not in certifi, so without this every
    # apismx/apipmx call fails with CERTIFICATE_VERIFY_FAILED. See tls_trust.
    tls_trust.install(settings)

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

    registry = TaggerRegistry()
    registry.discover("taggers.project")
    registry.discover("taggers.question")
    logger.info("taggers_registered", extra={"count": len(registry.all_taggers)})

    llm_client = build_llm_client(settings, skip_llm=skip_llm)

    # Bound the LLM response cache. It is written on every miss and never read
    # again once a prompt version moves on, so without this `.cache/` grows for
    # the life of the deployment — `log_retention` covers the two log sinks only.
    if not settings.skip_llm:
        from llm.cache import LLMCache
        LLMCache.prune(Path(settings.cache_dir), settings.llm_cache_retention_days)

    return AppContext(
        settings=settings,
        taxonomy=taxonomy,
        registry=registry,
        change_detector=ChangeDetector(Path(settings.cache_dir)),
        llm_client=llm_client,
    )


# Environment names that are NOT developer sandboxes. Matched as substrings of
# APP_ENV, which is spelled variously ("qauc", "sogo-beta", "live-uc").
_PROTECTED_ENVS = ("qauc", "beta", "live", "prod")


def _check_deployment_safety(settings: Settings) -> None:
    """Refuse to boot with a dev-only auth shortcut in a shared environment.

    `dev_auth_bypass` accepts a bare corp number as the whole credential, which is
    every tenant's data behind a guessable integer. Its own docstring says it must
    be False outside dev, and a comment cannot enforce that — this can.

    `auth_enabled=False` is only warned about: auth is deliberately parked until
    product integration, so refusing to boot on it would take down the current
    deployments. The warning is what makes the state visible in app.log.
    """
    app_env = os.environ.get("APP_ENV", "").strip().lower()
    protected = any(name in app_env for name in _PROTECTED_ENVS)

    if settings.dev_auth_bypass and protected:
        raise RuntimeError(
            f"SURVEY_TAGGER_DEV_AUTH_BYPASS is true with APP_ENV={app_env!r}. "
            "That accepts a bare corp number as the entire credential and must "
            "never be set outside a developer environment."
        )
    if not settings.auth_enabled:
        logger.warning(
            "auth_disabled_open_api",
            extra={"app_env": app_env or "(unset)", "protected_env": protected},
        )


def build_orchestrator(ctx: AppContext):
    """Construct the tenant-level PipelineOrchestrator from a context."""
    from pipeline.orchestrator import PipelineOrchestrator
    return PipelineOrchestrator(
        settings=ctx.settings,
        registry=ctx.registry,
        taxonomy=ctx.taxonomy,
        change_detector=ctx.change_detector,
        llm_client=ctx.llm_client,
    )
