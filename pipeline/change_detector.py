"""Change detection: skip unchanged work on re-runs.

Hashes the full set of inputs that affect a survey's tags — not just
survey_structure.json but the related/associated data too:

  Survey-level : survey_structure.json (content) + everything else in the
                 survey dir (response batch_*.parquet, invitations, linking,
                 prepop) by (size, mtime).
  Tenant-level : Directory/ + the Parallel.ai tenant_profile artifacts. These
                 feed the canon / journey tags, so a tenant_profile change
                 re-tags every survey.

A separate tenant hash covers the inputs to tenant_tags.json (directory +
profile). Thread-safe: the survey-hash file is guarded by a lock so parallel
tenant tagging can mark surveys processed concurrently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import sharefs
logger = logging.getLogger(__name__)

# Files small enough / semantic enough to hash by content. Everything else
# (parquet, blobs) is fingerprinted by (size, mtime) which is cheap and enough
# to detect a rewrite.
_CONTENT_HASH_SUFFIXES = {".json"}

# Files the pipeline itself writes into a directory it also fingerprints.
# `tagged_output.json` now lives inside the survey dir, beside its own inputs;
# without this exclusion writing the output would change the survey's input
# hash and every survey would re-tag on every run, forever.
_GENERATED_SURVEY_FILES = {"survey_structure.json", "tagged_output.json"}


def _short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _file_token(path: Path) -> str:
    """A change-sensitive token for one file."""
    try:
        if path.suffix.lower() in _CONTENT_HASH_SUFFIXES:
            return f"{path.name}:c:{hashlib.sha256(sharefs.read_bytes(path)).hexdigest()[:16]}"
        st = sharefs.stat(path)
        return f"{path.name}:m:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{path.name}:missing"


def _dir_token(directory: Path, *, exclude: set[str] | None = None) -> str:
    """Fingerprint every file under a directory (recursively), order-stable.

    Dotfiles are skipped: the atomic writers stage `.{name}.<rand>.tmp` files in
    the destination directory, and a leftover temp from a failed rename (more
    likely over SMB than locally) would otherwise change the hash forever.
    """
    if not sharefs.exists(directory):
        return f"{directory.name}:absent"
    exclude = exclude or set()
    tokens: list[str] = []
    for p in sorted(sharefs.rglob_files(directory)):
        if p.name.startswith(".") or p.name in exclude:
            continue
        tokens.append(str(p.relative_to(directory)) + "=" + _file_token(p))
    return f"{directory.name}:[" + ",".join(tokens) + "]"


class ChangeDetector:
    """Tracks input hashes to detect changed surveys / tenants between runs."""

    def __init__(self, cache_dir: Path) -> None:
        self.hash_file = cache_dir / "survey_hashes.json"
        self._hashes: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if self.hash_file.exists():
            try:
                with open(self.hash_file, "r", encoding="utf-8") as f:
                    self._hashes = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._hashes = {}

    def _save(self) -> None:
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, "w", encoding="utf-8") as f:
            json.dump(self._hashes, f, indent=2)

    @staticmethod
    def _survey_key(tenant_id: int, survey_no: int) -> str:
        return f"{tenant_id}/{survey_no}"

    @staticmethod
    def _tenant_key(tenant_id: int) -> str:
        return f"tenant:{tenant_id}"

    # ---------- hashing ----------

    def compute_hash(self, survey_dir: Path) -> str:
        """Back-compat: hash of survey_structure.json content only."""
        structure_file = survey_dir / "survey_structure.json"
        if not sharefs.exists(structure_file):
            return ""
        return hashlib.sha256(sharefs.read_bytes(structure_file)).hexdigest()[:16]

    def compute_survey_hash(
        self,
        survey_dir: Path,
        tenant_dir: Path | None = None,
        output_dir: Path | None = None,
        tenant_id: int | None = None,
        tenant_hash: str | None = None,
    ) -> str:
        """Composite hash over all inputs that affect this survey's tags.

        `tenant_hash` short-circuits the tenant half. Pass it whenever you are
        walking more than one survey of the same tenant: the tenant hash is
        identical for every one of them, but computing it recursively walks
        `Directory/` and content-reads every `tenant_profile/*.json` — the
        largest files on the share. Recomputed per survey (twice per processed
        survey, via `is_unchanged` then `mark_processed`) that is 2N redundant
        walks for an N-survey tenant, each one a batch of SMB round trips at
        ~230ms apiece on the production image server.
        """
        parts = [
            _file_token(survey_dir / "survey_structure.json"),
            _dir_token(survey_dir, exclude=_GENERATED_SURVEY_FILES),
        ]
        if tenant_hash is not None:
            parts.append(tenant_hash)
        elif tenant_dir is not None and tenant_id is not None:
            parts.append(self.compute_tenant_hash(tenant_dir, output_dir, tenant_id))
        return _short("|".join(parts))

    def compute_tenant_hash(
        self,
        tenant_dir: Path,
        output_dir: Path | None,
        tenant_id: int,
    ) -> str:
        """Hash over tenant-level inputs (directory + profile).

        The profile is read from `tenant_dir`, not `output_dir`: it is an input
        living beside CorporateData and Directory/ (see `settings.profile_root`).
        The two are the same path under SHARE_ROOT, which is why the old
        output_dir spelling worked in deployment — but locally it pointed at a
        directory that no longer holds profiles, so a profile change stopped
        invalidating the surveys that depend on it.

        `output_dir` is kept in the signature because callers pass it
        positionally; it is deliberately not consulted here.
        """
        parts = [
            _dir_token(tenant_dir / "Directory"),
            _dir_token(tenant_dir / "tenant_profile"),
        ]
        return _short("|".join(parts))

    # ---------- survey-level change API ----------

    def is_unchanged(
        self,
        tenant_id: int,
        survey_no: int,
        survey_dir: Path,
        tenant_dir: Path | None = None,
        output_dir: Path | None = None,
        tenant_hash: str | None = None,
    ) -> bool:
        """True when the survey's composite input hash matches the last run.

        `tenant_hash` — see `compute_survey_hash`. Pass the tenant's hash once
        per run rather than letting every survey recompute it.
        """
        key = self._survey_key(tenant_id, survey_no)
        current = self.compute_survey_hash(
            survey_dir, tenant_dir, output_dir, tenant_id, tenant_hash
        )
        with self._lock:
            stored = self._hashes.get(key)
        return stored is not None and current == stored

    def mark_processed(
        self,
        tenant_id: int,
        survey_no: int,
        survey_dir: Path,
        tenant_dir: Path | None = None,
        output_dir: Path | None = None,
        tenant_hash: str | None = None,
    ) -> None:
        """Record the survey's composite input hash after success.

        Must be given the SAME `tenant_hash` the matching `is_unchanged` used,
        or the survey is stored under a hash the next run will not reproduce and
        it re-tags forever. The orchestrator computes it once per tenant run and
        threads the one value into both calls.
        """
        key = self._survey_key(tenant_id, survey_no)
        current = self.compute_survey_hash(
            survey_dir, tenant_dir, output_dir, tenant_id, tenant_hash
        )
        with self._lock:
            self._hashes[key] = current
            self._save()

    def forget(self, tenant_id: int, survey_no: int) -> bool:
        """Drop a survey's stored hash so the next run re-tags it.

        Deleting `tagged_output.json` without this leaves the survey looking
        processed: the inputs still hash the same, so the run skips and the read
        route keeps 404ing. Returns whether an entry was actually removed.
        """
        key = self._survey_key(tenant_id, survey_no)
        with self._lock:
            existed = self._hashes.pop(key, None) is not None
            if existed:
                self._save()
        return existed

    # ---------- tenant-level change API (tenant_tags.json) ----------

    def tenant_is_unchanged(self, tenant_id: int, tenant_dir: Path, output_dir: Path) -> bool:
        key = self._tenant_key(tenant_id)
        current = self.compute_tenant_hash(tenant_dir, output_dir, tenant_id)
        with self._lock:
            stored = self._hashes.get(key)
        return stored is not None and current == stored

    def tenant_forget(self, tenant_id: int) -> bool:
        """Drop the tenant-level hash — same reason as `forget`, for
        `tenant_tags.json`."""
        key = self._tenant_key(tenant_id)
        with self._lock:
            existed = self._hashes.pop(key, None) is not None
            if existed:
                self._save()
        return existed

    def tenant_mark_processed(self, tenant_id: int, tenant_dir: Path, output_dir: Path) -> None:
        key = self._tenant_key(tenant_id)
        current = self.compute_tenant_hash(tenant_dir, output_dir, tenant_id)
        with self._lock:
            self._hashes[key] = current
            self._save()
