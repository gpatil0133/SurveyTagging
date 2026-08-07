"""Manual tenant-context overrides supplied per request.

The ad-hoc `POST /api/tag` path has no tenant on disk — no Parallel.ai
`tenant_profile`, no directory parquet, nothing. These five fields are the
caller's chance to hand the taggers the tenant context they'd otherwise read
from a profile, and they map 1:1 to the form inputs in `static/app.js`.

On the disk pipeline this is always empty: tenant context comes from
`TenantProfile` (the Parallel.ai org/cx/ex artifacts), so the deterministic
tiers that read these fields simply fall through.
"""

from pydantic import BaseModel


class ManualOverrides(BaseModel):
    """Caller-supplied tenant hints for the ad-hoc tagging endpoint."""

    industry: str = ""
    company_name: str = ""
    department: str = ""
    purpose: str = ""
    country: str = ""

    @property
    def is_empty(self) -> bool:
        """Whether the caller supplied no overrides at all."""
        return not any(
            (self.industry, self.company_name, self.department,
             self.purpose, self.country)
        )
