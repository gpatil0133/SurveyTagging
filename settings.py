"""Application settings loaded from environment variables and .env file."""

import os
import sys
from pathlib import Path, PureWindowsPath

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import sharefs


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SURVEY_TAGGER_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_model: str = "anthropic/claude-sonnet-4-6"
    llm_temperature: float = 0.1
    # V2: bumped from 4096 → 8192 to accommodate +6 new fields × up to 80 questions
    # in the Stage 5 question prompt without truncation.
    llm_max_tokens: int = 8192
    llm_rate_limit_rpm: int = 50
    # V2: enable Anthropic native prompt caching (ephemeral 5-min cache on
    # stable portions of prompts: system instructions, taxonomy enum lists,
    # industry stage lists). Saves ~90% input token cost on cache hits.
    llm_use_prompt_caching: bool = True

    # Canon embeddings (V5: tenant-canon + semantic stage scoring)
    # Local sentence-transformers model used to score per-question signatures
    # against the tenant canon. Lazy-loaded; cost paid once per process.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_top_k: int = 4
    # Question signatures whose top-1 candidate scores below this are bypassed
    # to a low_confidence_assigned status without an LLM call.
    embedding_min_score: float = 0.35
    # When the LLM agrees with a top candidate above this score AND the gap
    # to top-2 is comfortable, the assignment trends to high confidence.
    embedding_strong_score: float = 0.55

    # Pipeline
    #
    # `share_root` is the deployment setting: one root on the image-server /
    # network share holding both inputs and generated artifacts —
    #     {share_root}/{corp_no}/SurveyData/{survey_no}/survey_structure.json   (in)
    #     {share_root}/{corp_no}/SurveyData/{survey_no}/tagged_output.json      (out)
    #     {share_root}/{corp_no}/tenant_profile/*.json                          (in)
    #     {share_root}/{corp_no}/tenant_tags.json, tenant_canon_*.json          (out)
    # When set it overrides both `data_dir` and `output_dir` so the two can
    # never drift onto different anchors (see _derive_roots_from_share).
    #
    # Leave it unset for local dev / tests and set data_dir + output_dir
    # separately as before.
    #
    # Two accepted forms:
    #
    #   //172.16.1.105/sg-int-uc-img/AIChatbot   UNC — the process speaks SMB
    #                                            itself (see sharefs.py) using
    #                                            image_user / image_pass. No
    #                                            mount, so no root required.
    #   /mnt/aichatbot/AIChatbot                 a local path or CIFS mount
    #
    # Prefer forward slashes even for UNC. Backslashes are accepted and
    # normalized, but must be written UNQUOTED in .env — python-dotenv applies
    # unicode_escape to double-quoted values, collapsing the leading `\\` into
    # a single backslash.
    share_root: Path | None = None
    # Credentials for a UNC share_root; ignored for local/mounted roots.
    # `image_user` may carry a domain as DOMAIN\user or user@domain.
    # Accepted under either SURVEY_TAGGER_IMAGE_USER or a bare IMAGE_USER.
    image_user: str = Field(
        default="", validation_alias=AliasChoices("SURVEY_TAGGER_IMAGE_USER", "IMAGE_USER")
    )
    image_pass: str = Field(
        default="", validation_alias=AliasChoices("SURVEY_TAGGER_IMAGE_PASS", "IMAGE_PASS")
    )
    data_dir: Path = Field(default=Path("./extras/data"))
    output_dir: Path = Field(default=Path("./output"))
    # Stays LOCAL even when share_root is set — the LLM response cache and
    # survey_hashes.json are probed constantly and must not cost a network hop.
    cache_dir: Path = Field(default=Path("./.cache"))
    max_concurrent_surveys: int = 5
    skip_llm: bool = False

    # Auto-retag scheduler (periodic change-driven re-tagging).
    #
    # OFF by default. When enabled, a background task scans every tenant/survey
    # on an interval and re-tags only those whose inputs changed since the last
    # run (survey_structure + directory/response/invitation data +
    # tenant_profile). It calls the same service layer the manual retag
    # endpoints use — no separate tagging path. See scheduler.py.
    autoretag_enabled: bool = False
    autoretag_interval_minutes: int = 30
    # Force a full re-tag on every scan (ignore the change-detector). Normally
    # False so a scan is cheap and only touches changed inputs.
    autoretag_force: bool = False

    # Parallel.ai (tenant_profile fetcher) — used by `survey-tagger profile fetch`
    # to outsource website-driven org/CX/EX research. Read once per tenant
    # onboarding; not used by the per-survey tagging path.
    parallel_api_key: str = ""
    parallel_processor: str = "pro"          # ~10 min, blocking poll is fine
    parallel_api_timeout: int = 1800         # 30 min, generous ceiling for `pro`
    parallel_max_retries: int = 1            # extra attempts on empty/malformed agent output (each is a paid run)

    # Where tenant_profile artifacts come from.
    #
    #   "parallel" — this service calls Parallel.ai itself (parallel_* above)
    #                and owns the prompts in tenant_profile/prompts/.
    #   "smx"      — read profiles the SoGo Research API already generated
    #                (apismx /AIAccountProfile). Same three agents, so the
    #                artifacts written to the share are identical either way;
    #                no Parallel.ai key is needed in this mode.
    #
    # The on-disk contract does not change with this setting — only the producer.
    profile_source: str = "parallel"

    # Bearer token for apismx. Every Research API route requires one.
    #
    # Leave empty in a request-scoped context: the caller's own verified JWT is
    # forwarded instead (same issuer — see auth.py). This static value exists for
    # headless paths (CLI backfill, the auto-retag scheduler) that have no
    # inbound request to borrow a token from.
    smx_token: str = ""
    smx_request_timeout: float = 60.0

    # When a tenant has no profile on the share AND none in SMX, trigger
    # /AIAccountProfile/Generate and wait for it. Generation is a write that
    # starts paid research, so it is gated here as well as per-request.
    smx_allow_generate: bool = True
    # Post-generate polling. The observed run produced all three agents in ~44s;
    # the default window (6 x 15s) leaves generous headroom. Overrunning it is
    # not an error — generation continues server-side and the next fetch picks
    # it up from the /Details step.
    smx_generate_poll_attempts: int = 6
    smx_generate_poll_interval: float = 15.0

    # Logging
    log_level: str = "DEBUG"
    log_format: str = "console"

    # Observability. Both sinks live under `log_dir`, which stays LOCAL for the
    # same reason `cache_dir` does — these are written on every request and every
    # survey, and a network hop per line would show up in the numbers we are
    # trying to measure.
    #
    #   {log_dir}/app.log                    rotating text log (app + uvicorn)
    #   {log_dir}/usage-YYYY-MM-DD.jsonl     one JSON record per unit of work
    log_dir: Path = Field(default=Path("./logs"))
    # Write the JSONL usage/cost ledger. Off → the scopes still run, they just
    # emit nothing (no partial ledger to reason about).
    usage_log_enabled: bool = True
    # app.log rotation. 10 MB x 10 keeps roughly a week of DEBUG at current volume.
    app_log_max_bytes: int = 10 * 1024 * 1024
    app_log_backup_count: int = 10
    # Mirror app.log to stderr as well. False in a service host where stderr
    # goes nowhere; True in dev so `uvicorn --reload` still prints.
    log_to_stderr: bool = True

    # Per-million-token price overrides for cost accounting. Leave at 0.0 to use
    # litellm's built-in price map (which knows both Anthropic and OpenAI model
    # ids). Set them when running a model litellm prices as $0 — a negotiated
    # rate, a self-hosted endpoint, or a model newer than the installed litellm.
    llm_price_input_per_mtok: float = 0.0
    llm_price_output_per_mtok: float = 0.0

    # JWT (RS256, Research.Auth keypair) — see jwt-rs256-auth skill.
    # The PEM ships in the repo at survey_tagging/keys/public.pem.
    #
    # Auth is PARKED: `auth_enabled` is False by default so the API is open in
    # dev. Flip it (SURVEY_TAGGER_AUTH_ENABLED=true) at product integration —
    # the `require_auth` dependency in auth.py then enforces a Bearer JWT.
    auth_enabled: bool = False
    jwt_public_key_path: Path = Field(
        default=Path(__file__).parent / "keys" / "public.pem"
    )
    jwt_algorithm: str = "RS256"
    # DEV ONLY — must be False in intuc/qauc/beta/live. When true, the Bearer
    # value is treated as a literal numeric corp_no (no signature verification).
    dev_auth_bypass: bool = False

    # SoGo platform (outbound integration).
    #
    # Two ways to point this at a different SoGo environment:
    #
    # 1. Set `sogo_host` (recommended for env switches) — both apicx and
    #    apipmx URLs are derived from `https://<host>/apicx` and
    #    `https://<host>/apipmx`. Example values:
    #       intuc → sogolyticsintuc.sevenpv.com   (default)
    #       qauc  → sogoqauc.sogolytics.com
    #       beta  → sogobeta.sogolytics.com       (verify with SoGo team)
    #       prod  → sogo.sogolytics.com            (verify with SoGo team)
    #
    # 2. Override `sogo_apicx_base_url` / `sogo_apipmx_base_url` directly
    #    when the two services live on different hosts or behind different
    #    schemes. Setting either explicitly takes precedence over `sogo_host`.
    sogo_host: str = ""
    sogo_apicx_base_url: str = "https://sogolyticsintuc.sevenpv.com/apicx"
    sogo_apipmx_base_url: str = "https://sogolyticsintuc.sevenpv.com/apipmx"
    # Research API — /AIAccountProfile lives here (profile_source="smx").
    # apismx responses come back encrypted; apipmx /dcdata decrypts them, which
    # is why both URLs must track the same host.
    sogo_apismx_base_url: str = "https://sogolyticsintuc.sevenpv.com/apismx"
    sogo_request_timeout: float = 30.0
    sogo_max_retries: int = 3
    sogo_concurrency: int = 5

    # apicx endpoint paths (relative to sogo_apicx_base_url). Captured from
    # SoGo intuc's network trace — every endpoint here accepts the same
    # encrypted POST envelope handled by `_encrypted_post`. The PMX
    # (encrypt/decrypt) paths are not configurable — those are stable
    # platform-wide.
    sogo_category_create_path: str = "/AddEditDeleteCategory/detail/search"
    sogo_bulk_tag_path: str = "/SaveBulkAddEditDelete/detail/search"
    # CustomerJourney endpoint serves both `mode: UPDATE_JOURNEY` (read
    # tag/category catalogs and existing journey state) and
    # `mode: ADD_JOURNEY` (commit a new journey). The path is shared.
    sogo_customer_journey_path: str = "/CustomerJourney/detail/search"

    # Static service token forwarded to SoGo apicx when dev_auth_bypass=True.
    # Empty in production — the inbound caller's verified JWT is forwarded instead.
    sogo_dev_static_jwt: str = ""

    # TLS verification for outbound SoGo calls. Three-way pick:
    #
    #   1. `sogo_ca_bundle_path` set → use that PEM file. Production-clean.
    #      Use this when the SoGo host's cert chains to a corporate root CA
    #      that's installed in Windows' trust store but not in Python's
    #      bundled `certifi`. Export the corporate root as PEM and point
    #      this at it.
    #   2. `sogo_verify_ssl=False` → disable verification. Logged at
    #      WARNING on every client init. Acceptable for *intuc* dev only;
    #      MUST be true on qauc/beta/live.
    #   3. Default (verify_ssl=True, no bundle): httpx uses the certifi
    #      bundle. If the SoGo deployment uses a publicly-trusted cert
    #      (qauc, prod), this Just Works. If it uses an internal CA
    #      (intuc), this is what raises the "unable to get local issuer
    #      certificate" error you see in the logs.
    #
    # If `truststore` is installed (`pip install truststore`), the client
    # uses the OS trust store automatically — Windows will find the
    # corporate CA and verify without any config here.
    sogo_verify_ssl: bool = True
    sogo_ca_bundle_path: str = ""

    @field_validator("share_root", mode="before")
    @classmethod
    def _blank_share_root_is_unset(cls, value: object) -> object:
        """`SURVEY_TAGGER_SHARE_ROOT=` (blank) must mean "not configured".

        Pydantic would otherwise coerce the empty string to `Path('.')`, which is
        not None and so takes over `data_dir` and `output_dir` below — silently
        repointing every read and write at the current working directory.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _derive_roots_from_share(self) -> "Settings":
        """One share root drives both the input and output trees.

        Keeping them on the same anchor matters on Windows: `Path.relative_to`
        (service.delete_tagged, the profile DELETE route) raises ValueError the
        moment two paths disagree on their anchor, and `\\\\host\\share` vs
        `\\\\ip\\share` count as different anchors even when they resolve to the
        same server.
        """
        if self.share_root is None:
            return self

        raw = str(self.share_root)

        if sharefs.is_unc(raw):
            # A UNC root is served over SMB from inside the process, so it is
            # valid on every platform. Canonicalize to `//server/share/...`:
            # PosixPath treats `\` as an ordinary character, so the backslash
            # spelling would be one meaningless component that nothing can join
            # onto. Judge shape with PureWindowsPath, whose UNC parsing is
            # platform-independent.
            unc = sharefs.normalize(raw)
            if len(sharefs.unc_parts(unc)) < 2:
                raise ValueError(
                    f"SURVEY_TAGGER_SHARE_ROOT={raw!r} names a server but no share. "
                    "A UNC root needs at least //server/share."
                )
            if not self.image_user:
                raise ValueError(
                    f"SURVEY_TAGGER_SHARE_ROOT={raw!r} is a UNC path, which is "
                    "accessed over SMB and needs credentials, but "
                    "SURVEY_TAGGER_IMAGE_USER is empty. Set SURVEY_TAGGER_IMAGE_USER "
                    "and SURVEY_TAGGER_IMAGE_PASS in .env (a domain account is "
                    "written DOMAIN\\user)."
                )
            self.share_root = unc
        elif "\\" in raw:
            # Backslashes but not a UNC root — a drive-letter or root-relative
            # Windows path. Off Windows there is nothing to open it with, and a
            # UNC path that lost one of its leading backslashes (the .env
            # double-quoting trap) lands here too: it parses as *root-relative*
            # with an empty drive, so every read would silently resolve against
            # the current drive instead of the share.
            if not PureWindowsPath(raw).is_absolute():
                raise ValueError(
                    f"SURVEY_TAGGER_SHARE_ROOT={raw!r} is not an absolute path. A UNC "
                    r"root needs two leading backslashes (\\host\share\...). Write it "
                    "UNQUOTED in .env — python-dotenv unicode_escape-decodes "
                    r"double-quoted values, turning \\ into \."
                )
            if os.name != "nt":
                raise ValueError(
                    f"SURVEY_TAGGER_SHARE_ROOT={raw!r} is a Windows drive path, but "
                    f"this host is {sys.platform!r}, where it cannot be opened. "
                    "Use the UNC form to reach the share over SMB — e.g. "
                    "//172.16.1.105/sg-int-uc-img/AIChatbot — or a local mount point."
                )
        elif raw.startswith("/") and not self.share_root.is_absolute():
            # POSIX-style root on Windows: no drive and no UNC anchor, so it
            # resolves against whichever drive the process happens to be on.
            raise ValueError(
                f"SURVEY_TAGGER_SHARE_ROOT={raw!r} has no drive or UNC anchor and "
                "would resolve against the current drive. Give a fully anchored path."
            )

        self.data_dir = self.share_root
        self.output_dir = self.share_root
        return self

    @model_validator(mode="after")
    def _derive_sogo_urls_from_host(self) -> "Settings":
        """When `sogo_host` is set, rewrite any apicx/apipmx URL still on its
        default. Explicit per-URL overrides win — that's how to mix hosts
        (e.g. apicx on a private host, apipmx on the public one).
        """
        host = (self.sogo_host or "").strip().rstrip("/")
        if not host:
            return self

        # Strip any accidental scheme — we always emit https.
        for prefix in ("https://", "http://"):
            if host.lower().startswith(prefix):
                host = host[len(prefix):]
                break

        defaults = {
            "sogo_apicx_base_url": "https://sogolyticsintuc.sevenpv.com/apicx",
            "sogo_apipmx_base_url": "https://sogolyticsintuc.sevenpv.com/apipmx",
            "sogo_apismx_base_url": "https://sogolyticsintuc.sevenpv.com/apismx",
        }
        if self.sogo_apicx_base_url == defaults["sogo_apicx_base_url"]:
            self.sogo_apicx_base_url = f"https://{host}/apicx"
        if self.sogo_apipmx_base_url == defaults["sogo_apipmx_base_url"]:
            self.sogo_apipmx_base_url = f"https://{host}/apipmx"
        if self.sogo_apismx_base_url == defaults["sogo_apismx_base_url"]:
            self.sogo_apismx_base_url = f"https://{host}/apismx"
        return self

    @model_validator(mode="after")
    def _check_profile_source(self) -> "Settings":
        allowed = {"parallel", "smx"}
        value = (self.profile_source or "").strip().lower()
        if value not in allowed:
            raise ValueError(
                f"SURVEY_TAGGER_PROFILE_SOURCE={self.profile_source!r} is not one of "
                f"{sorted(allowed)}."
            )
        self.profile_source = value
        return self
