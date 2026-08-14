"""Main pipeline orchestrator: processes tenants and surveys."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import sharefs
import discovery
import usage_log
from config_loaders.industry_stages import IndustryStagesRegistry
from fs_utils import write_json_atomic
from llm.client import LLMClient
from llm.response_parser import ResponseParser
from loaders.context_assembler import assemble_context
from loaders.directory import load_directory_signals
from models.tags import TaggedSurvey, TagResult
from models.taxonomy import TaxonomyRegistry
from models.tenant_profile import TenantProfile
from pipeline.change_detector import ChangeDetector
from pipeline.single_survey import process_single_survey
from projections.tenant_tags_io import build_tenant_tags, write_tenant_tags
from settings import Settings
from taggers.registry import TaggerRegistry
from taggers.tenant import discover_tenant_taggers

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full tagging pipeline across tenants and surveys."""

    def __init__(
        self,
        settings: Settings,
        registry: TaggerRegistry,
        taxonomy: TaxonomyRegistry,
        llm_client: LLMClient | None = None,
        industry_stages: IndustryStagesRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.taxonomy = taxonomy
        self.llm_client = llm_client
        self.industry_stages = industry_stages
        self.change_detector = ChangeDetector(settings.cache_dir)
        self.response_parser = (
            ResponseParser(taxonomy, industry_stages) if taxonomy else None
        )
        # Per-thread LLM clients for bounded-parallel tenant tagging.
        self._tls = threading.local()

    def run(
        self,
        tenant_ids: list[int] | None = None,
        survey_nos: list[int] | None = None,
        force: bool = False,
    ) -> dict:
        """Run the pipeline synchronously.

        Args:
            tenant_ids: Specific tenants to process (None = all).
            survey_nos: Specific surveys to process (None = all).
            force: Ignore change detection, reprocess everything.

        Returns:
            Global summary dict.
        """
        data_dir = Path(self.settings.data_dir)
        config_dir = Path(__file__).parent.parent / "config"

        # Discover tenants
        tenants = self._discover_tenants(data_dir, tenant_ids)
        logger.info("pipeline_start", extra={"tenants": len(tenants)})

        global_stats = {
            "total_tenants": len(tenants),
            "total_surveys_processed": 0,
            "total_surveys_skipped": 0,
            "total_surveys_failed": 0,
            "tenant_summaries": {},
        }

        for tenant_id in tenants:
            tenant_dir = data_dir / str(tenant_id)
            tenant_stats = self._process_tenant(
                tenant_dir, tenant_id, survey_nos, config_dir, force
            )
            global_stats["tenant_summaries"][tenant_id] = tenant_stats
            global_stats["total_surveys_processed"] += tenant_stats["processed"]
            global_stats["total_surveys_skipped"] += tenant_stats["skipped"]
            global_stats["total_surveys_failed"] += tenant_stats["failed"]

        # Write the run summary to the local cache dir, not output_dir — the
        # latter is the share root, where a stray file would sit next to every
        # tenant folder.
        write_json_atomic(
            Path(self.settings.cache_dir) / "global_summary.json", global_stats
        )

        logger.info("pipeline_complete", extra=global_stats)
        return global_stats

    def _discover_tenants(self, data_dir: Path, tenant_ids: list[int] | None) -> list[int]:
        """Find tenant directories."""
        if tenant_ids:
            return tenant_ids

        tenants = []
        for item in sorted(sharefs.iterdir(data_dir)):
            if sharefs.is_dir(item):
                try:
                    tenant_id = int(item.name)
                    if sharefs.exists(item / "SurveyData"):
                        tenants.append(tenant_id)
                except ValueError:
                    continue
        return tenants

    def _process_tenant(
        self,
        tenant_dir: Path,
        tenant_id: int,
        survey_nos: list[int] | None,
        config_dir: Path,
        force: bool,
    ) -> dict:
        """Process all surveys for a single tenant."""
        logger.info("tenant_start", extra={"tenant_id": tenant_id})

        # Load tenant-level data (cached for all surveys)
        dir_signals = load_directory_signals(tenant_dir, config_dir)
        # Parallel.ai-derived profile (org/cx/ex artifacts). May be None when the
        # fetcher hasn't been run for this tenant — downstream code falls back
        # to directory / survey signals in that case.
        tenant_profile = TenantProfile.load(tenant_id, self.settings.profile_root)
        if tenant_profile is not None:
            logger.info(
                "tenant_profile_loaded",
                extra={
                    "tenant_id": tenant_id,
                    "has_org": tenant_profile.has_org,
                    "has_cx": tenant_profile.has_cx,
                    "has_ex": tenant_profile.has_ex,
                },
            )

        # One `kind="tenant"` ledger record covers everything charged to the
        # tenant rather than to a survey: the canon-derivation LLM calls (up to
        # one per journey type) and the tenant taggers. It is closed BEFORE the
        # survey loop opens its own scopes, so the two never nest and each
        # survey's cost stays its own.
        with usage_log.scope("tenant", tenant_id=tenant_id, unit="tenant_tags"):
            # V8: build the journey source for BOTH types straight from the
            # tenant profile. No LLM call and no artifact — the right one is
            # selected per survey by project_type (EX surveys use the employee
            # lifecycle, everything else CX) inside the LLM enhancement step.
            profile_journey, journey_index = self._load_profile_journey(
                tenant_id, tenant_profile, "CX")
            profile_journey_ex, journey_index_ex = self._load_profile_journey(
                tenant_id, tenant_profile, "EX")
            usage_log.annotate(
                has_journey_cx=profile_journey is not None,
                has_journey_ex=profile_journey_ex is not None,
                has_profile=tenant_profile is not None,
            )

            # V6: produce tenant-shape tags (compliance_posture, workforce_signature,
            # key_cx_touchpoints, etc.) once per tenant. Writes
            # output/{tenant_id}/tenant_tags.json. Best-effort — a failure here must
            # not block per-survey processing.
            try:
                self._tag_tenant(tenant_id, tenant_profile)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "tenant_tags_failed",
                    extra={"tenant_id": tenant_id, "error": str(e)},
                )
                # Swallowed for control flow, but the ledger should not call this
                # tenant a clean success.
                usage_log.set_status("partial")
                usage_log.annotate(tenant_tags_error=str(e))

        # Discover surveys
        survey_dir_base = tenant_dir / "SurveyData"
        if not sharefs.exists(survey_dir_base):
            return {"processed": 0, "skipped": 0, "failed": 0, "surveys": []}

        surveys = self._discover_surveys(survey_dir_base, survey_nos)

        # The tenant half of every survey's composite hash, computed ONCE for
        # the whole run. It recursively walks Directory/ and content-reads each
        # tenant_profile/*.json, so leaving it to the per-survey path cost 2N
        # walks of files that cannot change while the run is in flight (N for
        # the is_unchanged probes, N more for the mark_processed writes).
        # Threading one value in also guarantees the two agree: computing it
        # twice around a profile write would store a hash the next run cannot
        # reproduce, and the survey would re-tag forever.
        tenant_hash = self.change_detector.compute_tenant_hash(
            tenant_dir, Path(self.settings.output_dir), tenant_id
        )

        # Tag each survey. Bounded parallelism when max_concurrent_surveys > 1
        # AND there's more than one survey; otherwise sequential. The change
        # detector is thread-safe and per-survey output files are independent,
        # so parallel writes don't collide. Each worker thread gets its own LLM
        # client (the shared client's asyncio primitives can't cross loops).
        records = self._tag_surveys(
            surveys, survey_dir_base, tenant_dir, tenant_id,
            dir_signals, config_dir,
            tenant_profile, profile_journey, journey_index, force,
            profile_journey_ex, journey_index_ex,
            tenant_hash=tenant_hash,
        )

        stats = {"processed": 0, "skipped": 0, "failed": 0, "surveys": []}
        for rec in records:
            status = rec["status"]
            if status == "success":
                stats["processed"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
            stats["surveys"].append(rec)
        return stats

    def _tag_surveys(
        self, surveys, survey_dir_base, tenant_dir, tenant_id,
        dir_signals, config_dir,
        tenant_profile, profile_journey, journey_index, force,
        profile_journey_ex=None, journey_index_ex=None,
        tenant_hash=None,
    ) -> list[dict]:
        """Dispatch survey tagging sequentially or in a bounded thread pool."""
        # Captured on THIS thread, which is still inside the request's context.
        # ThreadPoolExecutor hands workers raw callables and does not copy
        # context, so without this every survey tagged in parallel would land in
        # the ledger with an empty request_id and a tenant-wide batch could not
        # be reassembled from its parts.
        ambient = usage_log.snapshot()

        def work(survey_no):
            usage_log.restore(ambient)
            return self._tag_one_survey(
                survey_no, survey_dir_base, tenant_dir, tenant_id,
                dir_signals, config_dir,
                tenant_profile, profile_journey, journey_index, force,
                profile_journey_ex, journey_index_ex,
                tenant_hash=tenant_hash,
            )

        max_workers = max(1, int(self.settings.max_concurrent_surveys))
        if max_workers == 1 or len(surveys) <= 1:
            return [work(s) for s in surveys]

        # Pre-warm the embedding model on this (single) thread before fanning
        # out. The lazy load is now lock-guarded, but loading it once up front
        # keeps the torch meta-tensor materialization off the worker threads
        # entirely and avoids N-1 workers blocking on the first load.
        #
        # Under the API the lifespan already warmed it on a background thread,
        # so this is a no-op returning immediately. It still matters for the
        # headless paths (scheduler, tests, any direct orchestrator use) that
        # never run a lifespan.
        if journey_index is not None and not self.settings.skip_llm:
            from llm.embeddings import EmbeddingModel
            EmbeddingModel.warm(self.settings.embedding_model)

        from concurrent.futures import ThreadPoolExecutor
        logger.info("tenant_parallel_tagging",
                    extra={"tenant_id": tenant_id, "surveys": len(surveys),
                           "max_workers": max_workers})
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(work, surveys))

    def _tag_one_survey(
        self, survey_no, survey_dir_base, tenant_dir, tenant_id,
        dir_signals, config_dir,
        tenant_profile, profile_journey, journey_index, force,
        profile_journey_ex=None, journey_index_ex=None,
        tenant_hash=None,
    ) -> dict:
        """Tag one survey: change-check, process, write, mark. Returns a record.

        The whole body runs inside one `kind="survey"` ledger scope, so every
        LLM call the survey makes — however deep — is attributed to it, and the
        record is emitted on the skip and failure paths too. A survey that
        failed after its first LLM call still spent money.
        """
        survey_dir = survey_dir_base / str(survey_no)
        output_dir = Path(self.settings.output_dir)

        with usage_log.scope("survey", tenant_id=tenant_id, survey_no=survey_no,
                             forced=bool(force)):
            if not force and self.change_detector.is_unchanged(
                tenant_id, survey_no, survey_dir,
                tenant_dir=tenant_dir, output_dir=output_dir,
                tenant_hash=tenant_hash,
            ):
                # "Inputs unchanged" only justifies a skip while the *output* of
                # the last run still exists. It can vanish independently of the
                # hash — DELETE /tags, a hand-cleaned share, a restored cache —
                # and then the survey would skip forever while every read 404s.
                # One stat per skipped survey, against a check that already read
                # survey_structure.json and walked the survey dir.
                if sharefs.exists(
                    discovery.tagged_output_path(output_dir, tenant_id, survey_no)
                ):
                    logger.info("survey_unchanged",
                                extra={"tenant": tenant_id, "survey": survey_no})
                    usage_log.set_status("skipped")
                    return {"survey_no": survey_no, "status": "skipped"}
                logger.info("survey_unchanged_but_output_missing_retagging",
                            extra={"tenant": tenant_id, "survey": survey_no})

            try:
                result = self._process_survey(
                    tenant_dir, tenant_id, survey_no, dir_signals, config_dir,
                    tenant_profile, profile_journey=profile_journey,
                    journey_index=journey_index,
                    profile_journey_ex=profile_journey_ex,
                    journey_index_ex=journey_index_ex,
                    llm_client=self._worker_llm_client(),
                )
                output_path = self._write_output(tenant_id, survey_no, result)
                # Only mark processed AFTER tagged_output.json is on disk, else the
                # survey re-tags next run instead of being silently skipped.
                if output_path is not None and sharefs.exists(output_path):
                    self.change_detector.mark_processed(
                        tenant_id, survey_no, survey_dir,
                        tenant_dir=tenant_dir, output_dir=output_dir,
                        tenant_hash=tenant_hash,
                    )
                else:
                    logger.warning(
                        "tagged_output_missing_after_write_skip_mark_processed",
                        extra={"tenant": tenant_id, "survey": survey_no,
                               "expected_path": str(output_path)},
                    )
                return {"survey_no": survey_no, "status": "success"}
            except Exception as e:  # noqa: BLE001
                logger.error("survey_failed",
                             extra={"tenant": tenant_id, "survey": survey_no, "error": str(e)})
                # Caught here, so the scope's own except path never sees it —
                # set the outcome explicitly.
                usage_log.set_status("failed")
                usage_log.annotate(error=f"{type(e).__name__}: {e}")
                return {"survey_no": survey_no, "status": "failed", "error": str(e)}

    def _worker_llm_client(self):
        """Per-thread LLM client. Sequential runs reuse the shared client; each
        worker thread builds its own so the client's asyncio primitives stay
        bound to that thread's event loop."""
        if self.llm_client is None:
            return None
        import threading
        max_workers = max(1, int(self.settings.max_concurrent_surveys))
        if max_workers == 1:
            return self.llm_client
        tls = self._tls
        client = getattr(tls, "llm_client", None)
        if client is None:
            from bootstrap import build_llm_client
            client = build_llm_client(self.settings) or self.llm_client
            tls.llm_client = client
        return client

    def _tag_tenant(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> None:
        """Run all tenant-level taggers and persist `tenant_tags.json`.

        Tenant tags are produced once per tenant and do not go through the
        per-survey registry/accumulator pipeline. Each tagger reads
        `tenant_profile` and emits one `TagResult`.
        """
        taggers = discover_tenant_taggers()
        results: dict[str, TagResult] = {}
        for tagger in taggers:
            try:
                results[tagger.tag_dimension] = tagger.tag(
                    tenant_id=tenant_id,
                    tenant_profile=tenant_profile,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "tenant_tagger_failed",
                    extra={"tagger": tagger.name, "tenant_id": tenant_id, "error": str(e)},
                )

        if not results:
            logger.info("tenant_tags_no_results", extra={"tenant_id": tenant_id})
            return

        artifact = build_tenant_tags(tenant_id, results, tenant_profile)
        write_tenant_tags(artifact, Path(self.settings.output_dir))

    def tag_tenant_only(self, tenant_id: int) -> dict | None:
        """Build + persist tenant_tags.json for one tenant (no survey tagging).

        Loads the tenant's Parallel.ai profile, runs the tenant-level taggers,
        writes the artifact, and returns it as a dict (None when no tenant tags
        were produced).
        """
        from projections.tenant_tags_io import load_tenant_tags

        with usage_log.scope("tenant", tenant_id=tenant_id, unit="tenant_tags_only"):
            tenant_dir = Path(self.settings.data_dir) / str(tenant_id)
            tenant_profile = TenantProfile.load(tenant_id, self.settings.profile_root)
            usage_log.annotate(has_profile=tenant_profile is not None)
            self._tag_tenant(tenant_id, tenant_profile)
            self.change_detector.tenant_mark_processed(
                tenant_id, tenant_dir, Path(self.settings.output_dir)
            )
            artifact = load_tenant_tags(tenant_id, Path(self.settings.output_dir))
            return artifact.model_dump() if artifact else None

    def _load_profile_journey(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
        journey_type: str,
    ):
        """Build the journey source + embedding index for one journey type.

        `journey_type` is "CX" or "EX". Returns
        (ProfileJourney | None, JourneyIndex | None).

        Reads `tenant_profile/` directly — no LLM call, no canon artifact, no
        share write. Both are None when the tenant has no profile for this
        journey type, which leaves `journey_stage` / `sub_stage_name`
        unassigned for the tenant's surveys. That is the intended behaviour:
        the previous industry-template fallback produced generic stage names
        that read as tenant-grounded and were not.

        Best-effort — any failure logs and returns (None, None) rather than
        failing the tenant run, since every other dimension is unaffected.
        """
        try:
            from llm.profile_journey import build_journey_index, build_profile_journey

            journey = build_profile_journey(tenant_id, tenant_profile, journey_type)
            if journey is None:
                logger.info(
                    "profile_journey_unavailable",
                    extra={"tenant_id": tenant_id, "journey_type": journey_type,
                           "has_profile": tenant_profile is not None},
                )
                return None, None

            if self.settings.skip_llm:
                # The index only exists to rank candidates for the question LLM
                # call. Without that call, embedding the leaves is pure cost.
                return journey, None

            # Embedding failure is scoped separately on purpose: the journey is
            # still real, and reporting "this tenant has no journey" when the
            # truth is "the embedding model would not load" sends whoever reads
            # the evidence to the wrong system.
            try:
                from llm.embeddings import EmbeddingModel

                embedder = EmbeddingModel.get(self.settings.embedding_model)
                return journey, build_journey_index(journey, embedder)
            except Exception as e:  # noqa: BLE001
                logger.warning("journey_index_build_failed",
                               extra={"tenant_id": tenant_id, "journey_type": journey_type,
                                      "leaves": len(journey.leaves), "error": str(e)})
                return journey, None
        except Exception as e:  # noqa: BLE001
            logger.warning("profile_journey_build_failed",
                           extra={"tenant_id": tenant_id, "journey_type": journey_type,
                                  "error": str(e)})
            return None, None

    def _discover_surveys(self, survey_dir_base: Path, survey_nos: list[int] | None) -> list[int]:
        """Find survey directories."""
        if survey_nos:
            return survey_nos

        surveys = []
        for item in sorted(sharefs.iterdir(survey_dir_base)):
            if sharefs.is_dir(item) and sharefs.exists(item / "survey_structure.json"):
                try:
                    surveys.append(int(item.name))
                except ValueError:
                    continue
        return surveys

    def _process_survey(
        self,
        tenant_dir: Path,
        tenant_id: int,
        survey_no: int,
        dir_signals,
        config_dir: Path,
        tenant_profile: TenantProfile | None = None,
        profile_journey=None,
        journey_index=None,
        profile_journey_ex=None,
        journey_index_ex=None,
        llm_client=None,
    ) -> TaggedSurvey:
        """Process a single survey through all pipeline stages.

        Context assembly (tenant-scoped) is the orchestrator's job; the actual
        tagging is delegated to the shared per-survey engine so single-survey and
        tenant runs produce identical output. `llm_client` overrides the shared
        client (used by parallel workers for their thread-local client).
        """
        # Stage 0: Assemble context
        logger.debug("survey_process_start",
                     extra={"tenant": tenant_id, "survey": survey_no})
        context = assemble_context(
            tenant_dir, survey_no, tenant_id, dir_signals, config_dir,
            tenant_profile=tenant_profile,
            profile_journey=profile_journey,
            journey_index=journey_index,
            profile_journey_ex=profile_journey_ex,
            journey_index_ex=journey_index_ex,
        )
        logger.debug(
            "context_assembled",
            extra={
                "survey": survey_no,
                "survey_name": context.survey_meta.title,
                "questions": len(context.questions),
                "non_cm_questions": len(context.non_cm_questions),
                "has_responses": context.has_responses,
                "has_profile_journey": context.profile_journey is not None,
            },
        )
        # Enough shape on the ledger record to explain an outlier cost without
        # having to open tagged_output.json.
        usage_log.annotate(
            survey_title=context.survey_meta.title,
            questions=len(context.questions),
            non_cm_questions=len(context.non_cm_questions),
        )

        return process_single_survey(
            context,
            self.registry,
            self.taxonomy,
            llm_client=llm_client if llm_client is not None else self.llm_client,
            response_parser=self.response_parser,
            industry_stages=self.industry_stages,
            settings=self.settings,
        )

    def _write_output(self, tenant_id: int, survey_no: int, tagged: TaggedSurvey) -> Path:
        """Write tagged output beside the survey's own inputs. Returns the path."""
        output_file = discovery.tagged_output_path(
            Path(self.settings.output_dir), tenant_id, survey_no
        )
        write_json_atomic(output_file, tagged.model_dump())

        logger.info("output_written", extra={
            "path": str(output_file),
            "tenant": tenant_id,
            "survey": survey_no,
        })
        return output_file
