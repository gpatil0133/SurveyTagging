"""SHA-256 hash-based disk cache for LLM responses.

V6: cache key now uses **per-prompt** versions sourced from each prompt YAML in
`config/prompts/`. Bumping the `version:` field in one YAML invalidates only
that prompt's cache — finer-grained than the previous global PROMPT_VERSION.

The `PROMPT_VERSION` constant below is kept as a backstop default for callers
that don't yet plumb through a per-prompt version (and as a documentation
anchor for the current major schema baseline).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# Default / fallback version when a caller doesn't pass `prompt_version`.
# Per-prompt versions in `config/prompts/*.yaml` override this for the
# project / question tagging calls.
#
# 6.0 (V6): `category` dimension renamed to `project_type` across the taxonomy
#           and all prompts. Prompt text moved out of code into
#           `config/prompts/*.yaml`. Cache keys now embed per-prompt versions.
# 5.0 (V5): atomic per-question journey block, canon-embedding candidates.
PROMPT_VERSION = "6.0"


class LLMCache:
    """Cache LLM responses keyed by (survey_hash, call_type, prompt_version)."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, survey_hash: str, call_type: str, prompt_version: str) -> Path:
        # Filename ordering: hash → call_type → version. The version goes last
        # so `*_v{old}_*.json` patterns can be cleaned up easily when prompts
        # are retired.
        return self.cache_dir / f"{survey_hash}_{call_type}_v{prompt_version}.json"

    def get(
        self,
        survey_hash: str,
        call_type: str,
        prompt_version: str | None = None,
    ) -> dict | None:
        """Retrieve cached LLM response, or None if not cached.

        `prompt_version` defaults to the module-level `PROMPT_VERSION` for
        backwards compatibility. Pass the per-prompt YAML version when
        available so prompt-specific cache invalidation works correctly.
        """
        version = prompt_version or PROMPT_VERSION
        path = self._cache_path(survey_hash, call_type, version)
        if not path.exists():
            logger.debug("disk_cache_miss", extra={"path": str(path)})
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("disk_cache_hit", extra={"path": str(path)})
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("cache_read_error", extra={"path": str(path), "error": str(e)})
            return None

    def put(
        self,
        survey_hash: str,
        call_type: str,
        response: dict,
        prompt_version: str | None = None,
    ) -> None:
        """Store an LLM response in cache."""
        version = prompt_version or PROMPT_VERSION
        path = self._cache_path(survey_hash, call_type, version)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            logger.debug("disk_cache_write", extra={"path": str(path)})
        except IOError as e:
            logger.warning("cache_write_error", extra={"path": str(path), "error": str(e)})

    @staticmethod
    def prune(cache_dir: Path, days: int) -> int:
        """Delete cached responses older than `days`. Returns how many went.

        Called once per process from `bootstrap`, not per write: the cache is read
        on every LLM call and a stat-walk per call would cost more than the entries
        save. `days <= 0` disables it — nothing is deleted rather than everything,
        which is the safe reading of "unset".

        Never raises: a locked or vanished file (routine on Windows) skips and the
        next boot retries.
        """
        if days <= 0 or not cache_dir.exists():
            return 0
        cutoff = time.time() - days * 86400
        deleted = 0
        for path in cache_dir.glob("*.json"):
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                deleted += 1
            except OSError:
                continue
        if deleted:
            logger.info("llm_cache_pruned",
                        extra={"deleted": deleted, "older_than_days": days,
                               "cache_dir": str(cache_dir)})
        return deleted

    def invalidate(self, survey_hash: str) -> None:
        """Remove all cached responses for a survey hash (all prompt versions)."""
        for path in self.cache_dir.glob(f"{survey_hash}_*.json"):
            path.unlink(missing_ok=True)

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of survey structure content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
