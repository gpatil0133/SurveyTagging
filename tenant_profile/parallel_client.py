"""Thin wrapper around the parallel-web SDK for tenant_profile fetches.

We submit a Task API run with the agent prompt as the natural-language
`output_schema` instruction (placeholders pre-substituted). The Task API
agent does its own multi-hop research; we get back a JSON-shaped payload
in `result.output.content`.

This wrapper:
  - Loads the API key from settings (PARALLEL_API_KEY)
  - Substitutes Jinja-style `{{var}}` placeholders into the prompt
  - Submits the run, polls until completion, and returns parsed JSON
  - Raises ParallelClientError on transport / parse failures (caller decides)

We do NOT retry transient failures here — the runner layer decides whether
to retry, since its idempotency check (artifact-on-disk) governs whether a
retry is actually a retry or a fresh fetch.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ParallelClientError(Exception):
    """Anything that prevents us from getting a parsed JSON payload back."""


@dataclass(frozen=True)
class TaskRunMeta:
    """Metadata about a Parallel.ai run we want to persist alongside the result."""

    run_id: str
    processor: str
    submitted_at: str
    completed_at: str


@dataclass(frozen=True)
class ParallelResult:
    payload: dict[str, Any]
    raw_content: Any                  # what Parallel returned, pre-parse — useful for debugging
    meta: TaskRunMeta


_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    """Substitute `{{var}}` placeholders. Missing vars raise — fail loudly."""
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise ParallelClientError(
                f"prompt template references {{{{ {key} }}}} but no value was provided"
            )
        return str(variables[key])
    return _PLACEHOLDER_RE.sub(_sub, template)


def _parse_payload(raw: Any) -> dict[str, Any]:
    """Parallel may return a dict, JSON, escape-doubled JSON, or imperfectly-quoted JSON.

    LLM agents occasionally emit JSON with bugs (unescaped quotes inside string values,
    trailing commas, etc.). We try strict parsing first, then progressively-lenient
    fallbacks before giving up. Order matters: cheap exact parses first, expensive
    repair attempts last.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        # 1. Strict JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        stripped = raw.strip()
        # 2. Markdown code fences ```json ... ```
        if stripped.startswith("```"):
            inner = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
            inner = re.sub(r"\n```\s*$", "", inner)
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass
        # 3. Doubly-escaped JSON: '{\"key\":\"value\"}' — Parallel sometimes wraps
        # the model output as a JSON-encoded string. Unwrap via unicode_escape.
        unescaped: str | None = None
        if stripped.startswith('{\\"') or stripped.startswith('[\\"'):
            try:
                unescaped = stripped.encode("latin-1", errors="replace").decode("unicode_escape")
                return json.loads(unescaped)
            except (UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError):
                pass
        # 4. Python-repr: {'key': 'value'} via ast.literal_eval
        try:
            result = ast.literal_eval(stripped)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        # 5. Last resort: json-repair (handles unescaped quotes, trailing commas,
        # missing brackets — all common LLM mistakes). Try both the raw input and
        # the unescaped version (if step 3 produced one).
        try:
            from json_repair import repair_json
        except ImportError:
            repair_json = None  # type: ignore[assignment]
        if repair_json is not None:
            for candidate in (unescaped, stripped):
                if candidate is None:
                    continue
                try:
                    repaired = repair_json(candidate, return_objects=True)
                    if isinstance(repaired, dict) and repaired:
                        logger.warning(
                            "agent_payload_repaired",
                            extra={"strategy": "json_repair",
                                   "from_unescaped": candidate is unescaped},
                        )
                        return repaired
                except Exception:  # noqa: BLE001 — repair lib is best-effort
                    continue
        # All strategies failed — dump raw for inspection, then raise
        debug_path = _dump_raw_for_debug(raw)
        preview = raw[:300].replace("\n", "\\n")
        raise ParallelClientError(
            f"agent returned non-JSON, non-dict-repr content "
            f"(starts with: {preview!r}; full content saved to {debug_path})"
        )
    raise ParallelClientError(f"unexpected payload type from Parallel: {type(raw).__name__}")


def _dump_raw_for_debug(raw: str) -> str:
    """Persist unparseable raw content to disk so we can inspect what Parallel returned."""
    import tempfile
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fd, path = tempfile.mkstemp(prefix=f"parallel_raw_{ts}_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
    except Exception:  # noqa: BLE001 — best-effort debug dump
        return "<debug dump failed>"
    return path


class ParallelClient:
    """Thin wrapper around `parallel.Parallel`."""

    def __init__(
        self,
        api_key: str | None = None,
        processor: str = "pro",
        api_timeout: int = 1800,
        max_retries: int = 1,
    ) -> None:
        key = api_key or os.environ.get("PARALLEL_API_KEY", "")
        if not key:
            raise ParallelClientError(
                "PARALLEL_API_KEY not set. Configure via .env "
                "(SURVEY_TAGGER_PARALLEL_API_KEY=...) or environment."
            )
        try:
            from parallel import Parallel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ParallelClientError(
                "parallel-web SDK not installed. Run: pip install parallel-web>=0.5.0"
            ) from e

        self._client = Parallel(api_key=key)
        self.processor = processor
        self.api_timeout = api_timeout
        # Number of additional attempts on ParallelClientError. 0 = no retry,
        # 1 = up to 2 total attempts. Each attempt is an independent task_run
        # so cost and latency multiply — keep this small.
        self.max_retries = max_retries

    def run_task(self, prompt: str, input_payload: dict[str, Any]) -> ParallelResult:
        """Submit a Task API run and block until completion.

        Retries up to `self.max_retries` additional times on ParallelClientError
        (transient network errors, empty/malformed agent payloads). Each retry
        is a fresh task_run.create — Parallel.ai pro calls are pay-per-run, so
        keep retry counts low.

        Args:
            prompt: Fully-rendered natural-language task instruction (placeholders
                    already substituted). Becomes the `output_schema` instruction.
            input_payload: Structured input passed to Parallel as `input`.
                           Surfaced to the agent alongside the prompt.

        Returns:
            ParallelResult with parsed JSON payload + run metadata.

        Raises:
            ParallelClientError after all attempts have failed.
        """
        max_attempts = self.max_retries + 1
        last_error: ParallelClientError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._run_task_once(prompt, input_payload)
            except ParallelClientError as e:
                last_error = e
                if attempt < max_attempts:
                    logger.warning(
                        "parallel_run_failed_retrying",
                        extra={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error": str(e),
                        },
                    )
                    continue
                raise
        # Unreachable: loop always returns or raises. Defensive only.
        raise last_error or ParallelClientError("run_task exited without result")

    def _run_task_once(self, prompt: str, input_payload: dict[str, Any]) -> ParallelResult:
        """Single submission + poll + parse. Raises ParallelClientError on any failure."""
        from datetime import datetime, timezone

        submitted_at = datetime.now(timezone.utc).isoformat()
        try:
            task_run = self._client.task_run.create(
                input=json.dumps(input_payload) if input_payload else "{}",
                processor=self.processor,
                task_spec={"output_schema": prompt},
            )
        except Exception as e:  # noqa: BLE001 — SDK error hierarchy varies
            raise ParallelClientError(f"task_run.create failed: {e}") from e

        run_id = getattr(task_run, "run_id", None) or getattr(task_run, "id", None)
        if not run_id:
            raise ParallelClientError(f"no run_id on task_run response: {task_run!r}")

        logger.info("parallel_task_submitted", extra={"run_id": run_id, "processor": self.processor})

        try:
            run_result = self._client.task_run.result(run_id, api_timeout=self.api_timeout)
        except Exception as e:  # noqa: BLE001
            raise ParallelClientError(f"task_run.result polling failed for {run_id}: {e}") from e

        completed_at = datetime.now(timezone.utc).isoformat()
        output = getattr(run_result, "output", None)
        if output is None:
            raise ParallelClientError(f"run {run_id} returned no output object")
        raw_content = getattr(output, "content", None)
        if raw_content is None:
            raise ParallelClientError(f"run {run_id} returned no output.content")

        # Fast-fail on obviously-broken agent output before walking parse fallbacks.
        # Empty or whitespace-only content cannot be JSON; very short non-JSON-shaped
        # content (e.g. the 16-byte string "output_citations" we've seen leak from
        # the SDK) is also unrecoverable. Surface these as distinct errors so
        # retry/dump diagnostics make sense.
        if isinstance(raw_content, str):
            stripped = raw_content.strip()
            if not stripped:
                raise ParallelClientError(
                    f"run {run_id} returned empty output.content "
                    f"(len={len(raw_content)}, agent produced no JSON)"
                )
            if len(stripped) < 50 and not (
                stripped[0] in "{[" or stripped.startswith('"{') or stripped.startswith('"[')
            ):
                debug_path = _dump_raw_for_debug(raw_content)
                raise ParallelClientError(
                    f"run {run_id} returned non-JSON-shaped content "
                    f"(len={len(stripped)}, starts with: {stripped[:50]!r}; "
                    f"full content saved to {debug_path})"
                )

        payload = _parse_payload(raw_content)
        meta = TaskRunMeta(
            run_id=run_id,
            processor=self.processor,
            submitted_at=submitted_at,
            completed_at=completed_at,
        )
        return ParallelResult(payload=payload, raw_content=raw_content, meta=meta)
