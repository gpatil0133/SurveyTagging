"""Periodic auto-retag scheduler — env-gated, OFF by default.

When `SURVEY_TAGGER_AUTORETAG_ENABLED=true`, a background asyncio task scans
every tenant/survey on an interval and re-tags only those whose inputs changed
since the last run (via the change detector). It calls the same service layer
the manual endpoints use — there is no separate tagging path here.

The scan is also exposed via `POST /api/admin/autoretag/run-now` so an operator
can trigger it on demand even when the periodic loop is disabled.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import discovery
import service
from bootstrap import AppContext, build_orchestrator

logger = logging.getLogger("survey_tagging.scheduler")


class AutoRetagScheduler:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_scan: str | None = None
        self._last_result: dict | None = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if not self.settings.autoretag_enabled:
            logger.info("autoretag_disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("autoretag_started",
                    extra={"interval_minutes": self.settings.autoretag_interval_minutes})

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        interval = max(1, int(self.settings.autoretag_interval_minutes)) * 60
        while self._running:
            try:
                await self.run_once()
            except Exception as e:  # noqa: BLE001
                logger.exception("autoretag_scan_failed", extra={"error": str(e)})
            await asyncio.sleep(interval)

    # ---------- the scan ----------

    async def run_once(self) -> dict:
        """One change-driven scan over all tenants/surveys. Returns a summary."""
        result = await asyncio.to_thread(self._scan)
        self._last_result = result
        self._last_scan = result.get("scanned_at")
        return result

    def _scan(self) -> dict:
        import usage_log

        # A scan has no inbound request, so it mints its own correlation id —
        # otherwise every survey it re-tags lands in the ledger with a blank
        # request_id and a scan cannot be costed as a unit. When the scan was
        # triggered through /api/admin/autoretag/run-now the caller's id is
        # already in context (asyncio.to_thread copied it) and wins.
        request_id = usage_log.current_request_id() or f"autoretag-{usage_log.new_request_id()}"
        handle = usage_log.bind_request(request_id)
        try:
            return self._scan_inner()
        finally:
            usage_log.reset_request(handle)

    def _scan_inner(self) -> dict:
        from datetime import datetime, timezone

        data_dir = Path(self.settings.data_dir)
        output_dir = Path(self.settings.output_dir)
        force = bool(self.settings.autoretag_force)
        orch = build_orchestrator(self.ctx)
        cd = orch.change_detector

        retagged_surveys: list[dict] = []
        retagged_tenants: list[int] = []

        for tenant_id in discovery.list_tenant_ids(data_dir):
            tdir = discovery.tenant_dir(data_dir, tenant_id)

            # Tenant-level tags: re-tag when directory/profile inputs changed.
            if force or not cd.tenant_is_unchanged(tenant_id, tdir, output_dir):
                try:
                    service.tag_tenant_tags(self.ctx, tenant_id)
                    retagged_tenants.append(tenant_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning("autoretag_tenant_tags_failed",
                                   extra={"tenant_id": tenant_id, "error": str(e)})

            # Surveys: re-tag those whose composite inputs changed. The tenant
            # half of the hash is computed once for the whole tenant — the scan
            # walks every survey of every tenant, so recomputing it per survey
            # made the periodic scan the heaviest reader on the share.
            tenant_hash = cd.compute_tenant_hash(tdir, output_dir, tenant_id)
            changed = []
            for sno in discovery.list_survey_nos(data_dir, tenant_id):
                sdir = discovery.survey_dir(data_dir, tenant_id, sno)
                if force or not cd.is_unchanged(tenant_id, sno, sdir,
                                                tenant_dir=tdir, output_dir=output_dir,
                                                tenant_hash=tenant_hash):
                    changed.append(sno)
            if changed:
                # One orchestrator run covers the changed surveys (bounded-parallel).
                orch.run(tenant_ids=[tenant_id], survey_nos=changed, force=True)
                retagged_surveys.extend({"tenant_id": tenant_id, "survey_no": s} for s in changed)

        return {
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "force": force,
            "retagged_tenants": retagged_tenants,
            "retagged_surveys": retagged_surveys,
            "retagged_survey_count": len(retagged_surveys),
        }

    # ---------- status ----------

    def status(self) -> dict:
        return {
            "enabled": bool(self.settings.autoretag_enabled),
            "running": self._running,
            "interval_minutes": self.settings.autoretag_interval_minutes,
            "force": bool(self.settings.autoretag_force),
            "last_scan": self._last_scan,
            "last_result": self._last_result,
        }
