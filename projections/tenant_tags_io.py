"""Tenant-tags artifact I/O.

Persists the result of running the `taggers/tenant/*` taggers once per tenant
to `output/{tenant_id}/tenant_tags.json`. The backend dashboard writer reads
this file alongside per-survey tagged_output.json + directory_tags.json to
make tenant-shaped widget decisions.

Artifact shape:

{
  "schema_version": "1.0",
  "generated_at": "...",
  "tenant_id": 12345,
  "tags": {
    "compliance_posture": {"value": "HIPAA", "source": "deterministic",
                           "confidence": 0.95, "evidence": "..."},
    "key_cx_touchpoints": {"value": ["Booking", "Stay"], ...},
    ...
  },
  "metadata": {"has_org": true, "has_cx": true, "has_ex": false}
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import sharefs
from models.tags import TagResult, TenantTags
from models.tenant_profile import TenantProfile

logger = logging.getLogger(__name__)


def tenant_tags_path(tenant_id: int, output_dir: Path) -> Path:
    return output_dir / str(tenant_id) / "tenant_tags.json"


def build_tenant_tags(
    tenant_id: int,
    results: dict[str, TagResult],
    tenant_profile: TenantProfile | None,
) -> TenantTags:
    """Pack per-tagger TagResults into a TenantTags artifact ready for write.

    Mirrors the per-tag shape used in TaggedSurvey.project_tags so consumers
    can treat the formats uniformly.
    """
    serialized: dict[str, dict] = {}
    for dim, tag in results.items():
        entry: dict = {
            "value": tag.value,
            "source": tag.source,
            "confidence": tag.confidence,
        }
        if tag.evidence:
            entry["evidence"] = tag.evidence
        serialized[dim] = entry

    metadata: dict = {}
    if tenant_profile is not None:
        metadata["has_org"] = tenant_profile.has_org
        metadata["has_cx"] = tenant_profile.has_cx
        metadata["has_ex"] = tenant_profile.has_ex
        if tenant_profile.org_confidence:
            metadata["org_confidence"] = tenant_profile.org_confidence
        if tenant_profile.cx_confidence:
            metadata["cx_confidence"] = tenant_profile.cx_confidence
        if tenant_profile.ex_confidence:
            metadata["ex_confidence"] = tenant_profile.ex_confidence
    else:
        metadata["has_org"] = False
        metadata["has_cx"] = False
        metadata["has_ex"] = False

    return TenantTags(tenant_id=tenant_id, tags=serialized, metadata=metadata)


def write_tenant_tags(
    tenant_tags: TenantTags,
    output_dir: Path,
) -> Path:
    """Write tenant_tags.json. Returns the path on success."""
    path = tenant_tags_path(tenant_tags.tenant_id, output_dir)
    sharefs.mkdir(path.parent)
    with sharefs.open_file(path, "w", encoding="utf-8") as f:
        json.dump(tenant_tags.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    logger.info(
        "tenant_tags_written",
        extra={"tenant_id": tenant_tags.tenant_id, "path": str(path), "tag_count": len(tenant_tags.tags)},
    )
    return path


def load_tenant_tags(tenant_id: int, output_dir: Path) -> TenantTags | None:
    """Read tenant_tags.json. Returns None if absent or unreadable."""
    path = tenant_tags_path(tenant_id, output_dir)
    if not sharefs.exists(path):
        return None
    try:
        with sharefs.open_file(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return TenantTags(**raw)
    except (json.JSONDecodeError, ValueError, OSError, KeyError) as e:
        logger.warning("tenant_tags_load_failed", extra={"tenant_id": tenant_id, "error": str(e)})
        return None
