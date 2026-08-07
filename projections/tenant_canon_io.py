"""Tenant canon artifact I/O.

Two files persist per tenant per journey type under
`{output_dir}/{tenant_id}/`:

  tenant_canon_{cx|ex}.json            — the canon Pydantic dump
  tenant_canon_{cx|ex}.embeddings.npz  — companion embeddings (numpy)

The legacy `journey_stages_{cx|ex}.json` artifact still exists on disk for
older tenants. `lift_legacy_to_canon` synthesizes a minimal v5.0 canon from
that older shape so transition-period code paths keep working without a
re-onboard.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import sharefs
from models.tenant_canon import CanonStage, TenantCanon, JourneyType

logger = logging.getLogger(__name__)


# Default industry-template stage description map for canon fallback. Mirrors
# `projections.tenant_stages._STAGE_DESCRIPTIONS` so our template-only canon
# carries the same one-sentence descriptions the legacy fallback used.
_TEMPLATE_DESCRIPTIONS: dict[str, str] = {
    "Inquiry": "Initial research about services, doctors, or facilities.",
    "Admission": "Registration, intake, and arrival at the facility.",
    "Treatment": "Ongoing clinical care, nursing, and in-facility experience.",
    "Discharge": "Checkout, billing, and handover back to the patient.",
    "Follow-up": "Post-care outreach, appointments, and recovery tracking.",
    "Application": "Account or product application and approval.",
    "Onboarding": "First-use setup and activation.",
    "Active": "Ongoing account usage and day-to-day activity.",
    "Closure": "Account closure or product termination.",
    "Awareness": "Initial brand or product discovery.",
    "Consideration": "Research and evaluation against alternatives.",
    "Purchase": "Conversion and first transaction.",
    "Delivery": "Fulfilment, shipping, and receipt of the order.",
    "Post-Purchase": "Ownership period, returns, and service.",
    "Loyalty": "Repeat purchase, rewards, and advocacy.",
    "Research": "Initial discovery of destinations or properties.",
    "Booking": "Reservation and payment.",
    "Pre-Stay": "Pre-arrival communications and preparation.",
    "Stay": "On-property experience during the trip.",
    "Post-Stay": "Check-out, follow-up, and feedback requests.",
    "Trial": "Product trial or evaluation period.",
    "Adoption": "Routine product use and feature uptake.",
    "Expansion": "Upsell, additional seats, or new products.",
    "Renewal": "Contract renewal decision.",
    "Service": "Ongoing service delivery.",
    "Resolution": "Case resolution and outcome communication.",
    "Retention": "Continued engagement and renewal.",
    "Advocacy": "Referral, reviews, loyalty, and promotion.",
    "Attract": "Employer brand and awareness among prospective employees.",
    "Recruit": "Candidate experience through application and hiring.",
    "Onboard": "First-day through first-90-days ramp-up and integration.",
    "Engage": "Day-to-day engagement, recognition, and culture.",
    "Develop": "Learning, career growth, and performance development.",
    "Retain": "Continued commitment, manager effectiveness, and tenure.",
    "Exit": "Offboarding, alumni relationships, and boomerang potential.",
}


# ---------- Path helpers ----------


def tenant_canon_path(tenant_id: int, journey_type: JourneyType, output_dir: Path) -> Path:
    return Path(output_dir) / str(tenant_id) / f"tenant_canon_{journey_type.lower()}.json"


def tenant_canon_embeddings_path(tenant_id: int, journey_type: JourneyType, output_dir: Path) -> Path:
    return Path(output_dir) / str(tenant_id) / f"tenant_canon_{journey_type.lower()}.embeddings.npz"


def tenant_canon_lock_path(tenant_id: int, journey_type: JourneyType, output_dir: Path) -> Path:
    return tenant_canon_path(tenant_id, journey_type, output_dir).with_suffix(".json.lock")


# ---------- Read / write ----------


def load_tenant_canon(tenant_id: int, journey_type: JourneyType, output_dir: Path) -> TenantCanon | None:
    """Load the canon artifact from disk. Returns None if missing or unparseable."""
    path = tenant_canon_path(tenant_id, journey_type, Path(output_dir))
    if not sharefs.exists(path):
        return None
    try:
        data = json.loads(sharefs.read_text(path, encoding="utf-8"))
        return TenantCanon.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("tenant_canon_load_failed", extra={"path": str(path), "error": str(e)})
        return None


def save_tenant_canon(canon: TenantCanon, output_dir: Path) -> Path:
    """Persist the canon as JSON. Returns the written path."""
    path = tenant_canon_path(canon.tenant_id, canon.journey_type, Path(output_dir))
    sharefs.mkdir(path.parent)
    sharefs.write_text(path, canon.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------- Legacy lift ----------


def _slugify(text: str) -> str:
    out: list[str] = []
    prev_dash = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash and out:
            out.append("-")
            prev_dash = True
    return "".join(out).rstrip("-") or "stage"


def lift_legacy_to_canon(
    legacy: dict,
    tenant_id: int,
    journey_type: JourneyType,
    industry: str = "",
) -> TenantCanon | None:
    """Synthesize a v5.0 canon from a legacy `journey_stages_{cx|ex}.json` dict.

    Used in the transition period when a tenant has the older artifact but no
    modern canon. We preserve stage names and descriptions verbatim; synonyms
    and customer goals are blank (the legacy artifact didn't carry them).
    """
    if not isinstance(legacy, dict):
        return None
    stages_raw = legacy.get("stages") or []
    if not stages_raw:
        return None

    used_ids: set[str] = set()
    canon_stages: list[CanonStage] = []
    for s in stages_raw:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").strip()
        if not name:
            continue
        base_id = _slugify(name)
        cid = base_id
        i = 2
        while cid in used_ids:
            cid = f"{base_id}-{i}"
            i += 1
        used_ids.add(cid)
        canon_stages.append(CanonStage(
            canon_id=cid,
            name=name,
            description=str(s.get("description") or "").strip(),
        ))

    if not canon_stages:
        return None

    return TenantCanon(
        schema_version="1.0",
        tenant_id=tenant_id,
        journey_type=journey_type,
        journey_name=str(legacy.get("journey_name") or
                         ("Customer Journey" if journey_type == "CX" else "Employee Journey")),
        industry=industry,
        source="legacy_lifted",
        locked=True,
        derived_at=str(legacy.get("derived_at") or datetime.now(timezone.utc).isoformat()),
        confidence="synthesized",
        stages=canon_stages,
        input_hash="",
    )


# ---------- Lock helper ----------


class CanonLock:
    """Best-effort cross-process lock used during canon build.

    Two processes asking to build the same canon will serialize: the second
    waits up to `timeout_s` for the first to finish, then re-reads the artifact.
    """

    def __init__(self, lock_path: Path, timeout_s: float = 60.0):
        self.lock_path = lock_path
        self.timeout_s = timeout_s
        self._fh = None

    def __enter__(self) -> "CanonLock":
        import time
        sharefs.mkdir(self.lock_path.parent)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                # "x" is exclusive-create on both backends: locally it is
                # O_CREAT|O_EXCL, over SMB it is a CREATE dispositon the server
                # rejects if the file exists. Either way the create is the
                # atom that decides who holds the lock.
                self._fh = sharefs.open_file(self.lock_path, "xb")
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    logger.warning("canon_lock_timeout", extra={"path": str(self.lock_path)})
                    return self  # proceed without the lock; second writer wins
                time.sleep(0.5)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        try:
            sharefs.unlink(self.lock_path, missing_ok=True)
        except OSError:
            pass


# ---------- Industry-template materialization ----------


def materialize_industry_template(
    journey_type: JourneyType,
    industry: str,
    registry,  # IndustryStagesRegistry; type-erased to avoid circular import
) -> list[dict]:
    """Build a list of {name, description} from journey_stages.yaml templates.

    Used as the `industry_stage_template` argument to `build_tenant_canon`.
    Mirrors `projections.tenant_stages.fallback_default_stages` resolution
    order: industry-specific stage list, then `_cx_default`/`_ex_default`,
    then a hardcoded safety floor.
    """
    template: list[dict] = []

    if journey_type == "CX" and industry:
        names = registry.get_stages(industry, project_type="CX")
        if names and names != registry.get_stages(None):
            for n in names:
                template.append({"name": n, "description": _TEMPLATE_DESCRIPTIONS.get(n, "")})
            return template

    block = registry.get_default_canonical(journey_type)
    if block and block.get("stages"):
        for s in block["stages"]:
            if isinstance(s, dict) and s.get("name"):
                template.append({
                    "name": s["name"],
                    "description": s.get("description") or _TEMPLATE_DESCRIPTIONS.get(s["name"], ""),
                })
        if template:
            return template

    if journey_type == "CX":
        names = ["Awareness", "Consideration", "Purchase", "Onboarding",
                 "Service", "Retention", "Advocacy"]
    else:
        names = ["Attract", "Recruit", "Onboard", "Engage", "Develop", "Retain", "Exit"]
    return [{"name": n, "description": _TEMPLATE_DESCRIPTIONS.get(n, "")} for n in names]


# ---------- Get-or-build orchestration ----------


async def get_or_build_tenant_canon_async(
    *,
    tenant_id: int,
    journey_type: JourneyType,
    output_dir: Path,
    llm,  # LLMClient | None
    tenant_profile,  # TenantProfile | None
    industry: str,
    corporate_purpose: str,
    industry_stages_registry,  # IndustryStagesRegistry
    embedder=None,  # EmbeddingModel | None — when provided, embeddings are persisted too
    embedding_model_name: str = "",
    force: bool = False,
) -> TenantCanon:
    """Read canon from disk if present (and not force); else build, persist, return.

    When `embedder` is provided, the embeddings file is also (re)computed and
    persisted alongside the canon. Caller is expected to hold the lock around
    this call if multiple workers might race.
    """
    from llm.embeddings import build_index, save_embeddings  # local import to avoid cycle
    from llm.tenant_canon import build_tenant_canon, compute_canon_input_hash

    canon_path = tenant_canon_path(tenant_id, journey_type, Path(output_dir))

    template = materialize_industry_template(journey_type, industry, industry_stages_registry)

    existing = None if force else load_tenant_canon(tenant_id, journey_type, output_dir)
    if existing is not None and existing.stages:
        # A human-approved (locked) canon is authoritative — never rebuild.
        if existing.locked:
            return existing
        # Otherwise reuse the persisted canon only while its inputs are
        # unchanged. When the tenant profile lands or changes, the hash
        # differs and we fall through to rebuild — this is what stops a
        # profile-less first build from freezing the generic template.
        expected_hash = compute_canon_input_hash(
            tenant_profile=tenant_profile,
            journey_type=journey_type,
            industry_stage_template=template,
        )
        if existing.input_hash and existing.input_hash == expected_hash:
            return existing
        logger.info(
            "tenant_canon_inputs_changed_rebuilding",
            extra={
                "tenant_id": tenant_id, "journey_type": journey_type,
                "old_hash": existing.input_hash, "new_hash": expected_hash,
                "old_source": existing.source,
            },
        )

    canon = await build_tenant_canon(
        llm=llm,
        tenant_profile=tenant_profile,
        journey_type=journey_type,
        industry_stage_template=template,
        industry=industry,
        corporate_purpose=corporate_purpose,
        tenant_id=tenant_id,
        force=force,
    )

    # Downgrade guard: never silently replace an agent-derived canon with the
    # generic industry template. A profile that is temporarily unreadable (or
    # transiently absent) would otherwise wipe out a good agent canon. Keep the
    # richer existing canon unless the caller explicitly forced a rebuild.
    if (
        not force
        and canon.source == "industry_template"
        and existing is not None
        and existing.stages
        and existing.source in ("agent_canon", "agent_blended")
    ):
        logger.warning(
            "tenant_canon_downgrade_suppressed",
            extra={
                "tenant_id": tenant_id, "journey_type": journey_type,
                "existing_source": existing.source,
            },
        )
        return existing

    save_tenant_canon(canon, output_dir)

    if embedder is not None:
        try:
            index = build_index(canon, embedder)
            npz = tenant_canon_embeddings_path(tenant_id, journey_type, output_dir)
            save_embeddings(npz, index)
        except Exception as e:  # noqa: BLE001
            logger.warning("canon_embeddings_build_failed",
                           extra={"tenant_id": tenant_id, "error": str(e)})

    logger.info(
        "tenant_canon_persisted",
        extra={
            "tenant_id": tenant_id, "journey_type": journey_type,
            "source": canon.source, "stage_count": len(canon.stages),
        },
    )
    return canon
