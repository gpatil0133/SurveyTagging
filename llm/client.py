"""LLM client wrapper using litellm with retry, rate limiting, and caching.

V2 Features:
- Anthropic native prompt caching: `cache_control: {"type": "ephemeral"}` markers
  on stable system prompt content. 90%+ input-token cost reduction on cache hits.
- Structured message blocks (system + cached taxonomy preamble + dynamic user).
- Response caching (survey-level) via LLMCache for full-response reuse across re-runs.

Prompt cache lifecycle:
- Ephemeral (5-min TTL on Anthropic's side). Ideal for batch processing of
  many surveys within a single run — stable system prompt hits cache.
- First request in a batch is a "cache write" (slightly more expensive).
- Subsequent requests within TTL are "cache reads" at ~10% input token cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path

import usage_log
from llm.cache import LLMCache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _accepts_temperature(model: str, temperature: float) -> bool:
    """Will this model accept this temperature, or must the param be dropped?

    OpenAI's reasoning families (gpt-5*, o-series) accept `temperature=1` only,
    and litellm raises `UnsupportedParamsError` for anything else rather than
    silently correcting it — so a default of 0.1 that is fine on Anthropic kills
    every call on `openai/gpt-5-mini`.

    Asking litellm's own param mapper is local (no network, no key) and keeps
    the rule where it belongs: with the provider matrix, not in a model-name
    prefix list here that would need editing on every new family. A probe that
    fails for any other reason answers True — the real call then reports the
    real error instead of us quietly changing the request.
    """
    try:
        from litellm.utils import get_llm_provider, get_optional_params

        model_name, provider, _, _ = get_llm_provider(model=model)
        get_optional_params(
            model=model_name,
            custom_llm_provider=provider,
            temperature=temperature,
        )
        return True
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "UnsupportedParamsError":
            logger.info(
                "llm_temperature_dropped",
                extra={"model": model, "temperature": temperature,
                       "reason": str(e)[:200]},
            )
            return False
        logger.debug("llm_temperature_probe_failed",
                     extra={"model": model, "error": str(e)})
        return True


def _cache_tokens(usage) -> tuple[int, int]:
    """Pull (cache_read, cache_write) out of a litellm usage block.

    litellm's `Usage` model declares neither `cache_read_input_tokens` nor
    `cache_creation_input_tokens`; the canonical home is
    `usage.prompt_tokens_details.{cached_tokens, cache_creation_tokens}`, which
    **every** provider populates. The Anthropic-style attributes exist only
    because `Usage` sets `extra="allow"` and the Anthropic handler happens to
    pass those kwargs — the OpenAI handler does not.

    So reading the flat attributes alone silently reports zero on OpenAI, which
    makes prompt caching unmeasurable exactly when it is being tuned. Read the
    details block first, fall back to the Anthropic extras.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    read = getattr(details, "cached_tokens", None) if details is not None else None
    write = getattr(details, "cache_creation_tokens", None) if details is not None else None
    if read is None:
        read = getattr(usage, "cache_read_input_tokens", 0)
    if write is None:
        write = getattr(usage, "cache_creation_input_tokens", 0)
    return int(read or 0), int(write or 0)


class LLMClient:
    """Wraps litellm for LLM calls with retry, caching, and structured output parsing."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        rate_limit_rpm: int = 50,
        cache_dir: Path | None = None,
        use_prompt_caching: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_rpm = rate_limit_rpm
        self._semaphore = asyncio.Semaphore(max(1, rate_limit_rpm // 10))
        self.cache = LLMCache(cache_dir) if cache_dir else None
        self.use_prompt_caching = use_prompt_caching

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        cache_key: str | None = None,
        call_type: str = "general",
        cached_system_preamble: str | None = None,
        prompt_version: str | None = None,
    ) -> dict | None:
        """Call LLM with retry, optional response caching, and prompt caching.

        Args:
            prompt: User prompt content (dynamic per survey).
            system_prompt: System-level instructions (stable across surveys).
            cache_key: If provided, check/store full response in disk cache.
            call_type: Cache call type identifier (e.g., "project", "question").
            cached_system_preamble: Additional stable content (taxonomy enums,
                industry stage lists) that should be marked cache_control=ephemeral
                for Anthropic prompt caching. Goes before system_prompt.
            prompt_version: Per-prompt version string (from the prompt YAML).
                Threaded to the disk cache so bumping one prompt's version
                invalidates only its cache. Falls back to `cache.PROMPT_VERSION`
                when None.

        Returns:
            Parsed JSON dict from LLM response, or None on failure.
        """
        logger.debug(
            "llm_complete_start",
            extra={"call_type": call_type, "model": self.model,
                   "prompt_version": prompt_version, "cache_key": cache_key,
                   "prompt_chars": len(prompt),
                   "has_cached_preamble": bool(cached_system_preamble)},
        )

        # Check disk cache first (full-response cache)
        if self.cache and cache_key:
            cached = self.cache.get(cache_key, call_type, prompt_version=prompt_version)
            if cached is not None:
                logger.info("llm_response_cache_hit",
                            extra={"cache_key": cache_key, "call_type": call_type,
                                   "prompt_version": prompt_version})
                # Recorded, not skipped: the ledger must be able to explain why
                # a re-tagged survey cost $0 instead of looking like a run that
                # never called the model.
                usage_log.record_llm_call(
                    call_type=call_type, model=self.model,
                    cached=True, ok=True, cost_usd=0.0,
                    input_tokens=0, output_tokens=0,
                )
                return cached
            logger.debug("llm_response_cache_miss",
                         extra={"cache_key": cache_key, "call_type": call_type,
                                "prompt_version": prompt_version})

        last_error = None
        for attempt in range(3):
            try:
                async with self._semaphore:
                    response = await self._call_llm(
                        prompt, system_prompt, cached_system_preamble,
                        call_type=call_type, attempt=attempt + 1,
                    )

                if response is None:
                    logger.debug("llm_empty_response",
                                 extra={"attempt": attempt + 1, "call_type": call_type})
                    continue

                logger.debug("llm_response_received",
                             extra={"attempt": attempt + 1, "call_type": call_type,
                                    "response_chars": len(response)})

                parsed = self._extract_json(response)
                if parsed is not None:
                    if self.cache and cache_key:
                        self.cache.put(cache_key, call_type, parsed, prompt_version=prompt_version)
                    logger.debug("llm_response_parsed",
                                 extra={"call_type": call_type,
                                        "top_level_keys": list(parsed.keys())[:20]})
                    return parsed

                logger.warning("llm_json_parse_failed",
                               extra={"attempt": attempt + 1,
                                      "response_preview": response[:200]})

            except Exception as e:
                last_error = e
                logger.warning(
                    "llm_call_failed attempt=%d error=%s: %s",
                    attempt + 1, type(e).__name__, e,
                    extra={"attempt": attempt + 1, "error": str(e)},
                )
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        logger.error(
            "llm_all_retries_failed call_type=%s last_error=%s: %s",
            call_type, type(last_error).__name__ if last_error else "None", last_error,
            extra={"error": str(last_error), "call_type": call_type},
        )
        return None

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        cached_system_preamble: str | None = None,
        call_type: str = "general",
        attempt: int = 1,
    ) -> str | None:
        """Make the actual LLM API call via litellm with optional prompt caching.

        `call_type`/`attempt` are carried purely for the usage ledger — every
        provider round-trip, successful or not, is recorded here because this is
        the only frame where both the response usage block and the wall time of
        the round-trip exist.
        """
        try:
            import litellm

            # Build Anthropic-style system blocks with cache_control on stable content
            # litellm passes these through for Anthropic models.
            # For non-Anthropic models, fall back to a simple concatenated system string.
            is_anthropic = "claude" in self.model.lower() or "anthropic" in self.model.lower()

            if is_anthropic and self.use_prompt_caching and cached_system_preamble:
                # Structured system blocks: stable preamble is cache-controlled,
                # per-call system prompt is NOT cached.
                system_blocks = [
                    {
                        "type": "text",
                        "text": cached_system_preamble,
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
                if system_prompt:
                    system_blocks.append({
                        "type": "text",
                        "text": system_prompt,
                    })
                messages = [{"role": "user", "content": prompt}]
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "system": system_blocks,
                    "max_tokens": self.max_tokens,
                }
            else:
                # Fallback: simple system string
                messages = []
                combined_system = ""
                if cached_system_preamble:
                    combined_system = cached_system_preamble
                if system_prompt:
                    combined_system = (combined_system + "\n\n" + system_prompt).strip() \
                        if combined_system else system_prompt
                if combined_system:
                    messages.append({"role": "system", "content": combined_system})
                messages.append({"role": "user", "content": prompt})
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                }

            # Sent only when the provider will take it — see _accepts_temperature.
            # litellm maps max_tokens → max_completion_tokens itself where needed.
            send_temperature = _accepts_temperature(self.model, self.temperature)
            if send_temperature:
                kwargs["temperature"] = self.temperature

            logger.debug(
                "litellm_request",
                extra={"model": self.model, "is_anthropic": is_anthropic,
                       "prompt_caching": is_anthropic and self.use_prompt_caching
                       and bool(cached_system_preamble),
                       "max_tokens": self.max_tokens,
                       "temperature": self.temperature if send_temperature else None},
            )
            started = time.perf_counter()
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as e:
                usage_log.record_llm_call(
                    call_type=call_type, model=self.model, ok=False, attempt=attempt,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    error=f"{type(e).__name__}: {e}",
                )
                raise
            duration_ms = int((time.perf_counter() - started) * 1000)

            # Usage + cost. Never let accounting break a tagging run: a provider
            # that omits `usage`, or a model litellm cannot price, degrades to
            # nulls in the ledger rather than an exception here.
            in_tok = out_tok = cache_read = cache_write = 0
            try:
                usage = response.usage
                in_tok = getattr(usage, "prompt_tokens", 0) or 0
                out_tok = getattr(usage, "completion_tokens", 0) or 0
                cache_read, cache_write = _cache_tokens(usage)
                if cache_read or cache_write:
                    logger.info("llm_prompt_cache_stats",
                                extra={"cache_read_tokens": cache_read,
                                       "cache_write_tokens": cache_write,
                                       "input_tokens": in_tok,
                                       "output_tokens": out_tok})
            except Exception:  # noqa: BLE001
                pass

            usage_log.record_llm_call(
                call_type=call_type, model=self.model, ok=True, attempt=attempt,
                duration_ms=duration_ms,
                input_tokens=in_tok, output_tokens=out_tok,
                cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                cost_usd=usage_log.estimate_cost(response, in_tok, out_tok),
            )

            return response.choices[0].message.content

        except ImportError:
            logger.error("litellm_not_installed")
            return None
        except Exception as e:
            raise e

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract JSON object from LLM response text.

        Handles responses wrapped in markdown code blocks or plain JSON.
        """
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
