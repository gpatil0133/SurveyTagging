"""Admin routes — the auto-retag scheduler (scheduler.py).

Lifted out of run.py unchanged — see api/__init__.py for why the split."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("survey_tagging.api")

router = APIRouter()


# ====================================================================
# §6  Admin — the auto-retag scheduler (scheduler.py)
#     Separate from §5 because these are the only non-tenant routes that
#     *do* something: run-now kicks off a full change-scan.
# ====================================================================

@router.get("/api/admin/autoretag")
async def autoretag_status(request: Request) -> dict:
    """Auto-retag scheduler status (enabled?, interval, last scan).

    The scheduler is owned by the app's lifespan, so it is read off
    `request.app.state` rather than a module global — a router has no `app` of its
    own, and reaching back into run.py for one would be an import cycle.
    """
    sched = getattr(request.app.state, "scheduler", None)
    if sched is None:
        return {"enabled": False, "status": "uninitialized"}
    return sched.status()


@router.post("/api/admin/autoretag/run-now")
async def autoretag_run_now(request: Request) -> dict:
    """Run one change-scan immediately (works even when the periodic loop is off)."""
    sched = getattr(request.app.state, "scheduler", None)
    if sched is None:
        raise HTTPException(503, "Scheduler not initialized")
    return await sched.run_once()
