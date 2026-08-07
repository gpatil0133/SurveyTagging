"""Tenant profile fetcher — outsources website-driven org/CX/EX research to Parallel.ai.

Three artifacts persist per tenant under
    {output_dir}/{tenant_id}/tenant_profile/
        org_profile.json
        cx_intelligence.json
        ex_intelligence.json

The org agent runs first and feeds the CX and EX agents. All three are
idempotent — re-running a fetch is a no-op unless --force is passed.

This module is consumed only by the `survey-tagger profile *` CLI in this PR.
Phase 2+ will wire the artifacts into the tagging pipeline.
"""

from tenant_profile.runner import run_org, run_cx, run_ex, ArtifactExists, FetchResult
from tenant_profile.batch import run_batch, BatchResult
from tenant_profile.parallel_client import ParallelClient, ParallelClientError

__all__ = [
    "run_org",
    "run_cx",
    "run_ex",
    "run_batch",
    "ArtifactExists",
    "FetchResult",
    "BatchResult",
    "ParallelClient",
    "ParallelClientError",
]
