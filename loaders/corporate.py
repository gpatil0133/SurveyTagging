"""Load corporate/tenant context from {TenantID}_CorporateData.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models.corporate import CorporateContext

logger = logging.getLogger(__name__)


def load_corporate(tenant_dir: Path, tenant_id: int) -> CorporateContext:
    """Load corporate data for a tenant.

    Args:
        tenant_dir: Path to the tenant folder (e.g., .../75885/)
        tenant_id: Numeric tenant ID

    Returns:
        CorporateContext populated from JSON, or empty defaults if file missing.
    """
    corporate_file = tenant_dir / f"{tenant_id}_CorporateData.json"

    if not corporate_file.exists():
        logger.warning("corporate_file_missing", extra={"tenant_id": tenant_id})
        return CorporateContext()

    try:
        with open(corporate_file, "r", encoding="utf-8-sig", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("corporate_file_parse_error", extra={"tenant_id": tenant_id, "error": str(e)})
        return CorporateContext()

    if not data or not isinstance(data, list) or len(data) == 0:
        logger.warning("corporate_file_empty", extra={"tenant_id": tenant_id})
        return CorporateContext()

    raw = data[0]

    return CorporateContext(
        corporate_no=int(raw.get("corporate_no", 0) or 0),
        corporate_id=str(raw.get("corporate_id", "") or ""),
        corporate_name=str(raw.get("corporate_name", "") or ""),
        first_name=str(raw.get("first_name", "") or ""),
        last_name=str(raw.get("last_name", "") or ""),
        job_title=str(raw.get("job_title", "") or ""),
        email_address=str(raw.get("email_address", "") or ""),
        country=str(raw.get("country", "") or ""),
        country_name=str(raw.get("CountryName", "") or ""),
        state=str(raw.get("state", "") or ""),
        city=str(raw.get("city", "") or ""),
        industry=str(raw.get("Industry", "") or ""),
        department=str(raw.get("Department", "") or ""),
        purpose=str(raw.get("Purpose", "") or ""),
        corporate_size=str(raw.get("corporate_size", "") or ""),
        revenue=str(raw.get("Revenue", "") or ""),
        is_cx=bool(raw.get("isCX", False)),
        is_engage=bool(raw.get("isEngage", False)),
        is_wlc=bool(raw.get("isWLC", False)),
        is_enterprise_plus=bool(raw.get("isEnterprisePlus", False)),
        account_type=int(raw.get("account_type", 0) or 0),
        sub_account_type=int(raw.get("sub_account_type", 0) or 0),
        lang=str(raw.get("lang", "en") or "en"),
    )
