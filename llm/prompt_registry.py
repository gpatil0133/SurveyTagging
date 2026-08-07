"""Centralized prompt loader: reads YAML templates from `config/prompts/` and
renders them with Jinja2.

Why this exists
---------------
Prompt text is configuration, not code. Pulling it out of Python:

- Lets us iterate on prompts without redeploying the package
- Surfaces the cached-vs-dynamic split clearly (top-level YAML keys)
- Versions each prompt independently, so changing one prompt invalidates
  only its on-disk LLM response cache (see `llm.cache`)
- Keeps the **stable** portion (instructions + taxonomy enums + canonical
  lists) maximal so the Anthropic ephemeral prompt cache hits hard:
  cache write once per process, cache reads on every subsequent survey.

YAML schema (per file)
----------------------
```yaml
version: "6.0"            # bumped per-prompt, used as cache-key component
description: "..."        # human-readable, for docs/UI
cached_preamble: |        # Jinja template — rendered with `cached_context`
  ...                     #   ↑ goes into Anthropic cache_control=ephemeral
user_prompt: |            # Jinja template — rendered with `user_context`
  ...                     #   ↑ dynamic per call, never cached
```

The split between `cached_context` and `user_context` is the cache contract:
inputs to `cached_preamble` must be stable across all calls within a run
(taxonomy enums, dashboard lists, journey rules). Inputs to `user_prompt`
are per-survey.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import yaml
from jinja2 import Environment, StrictUndefined, Template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderedPrompt:
    """Output of `PromptRegistry.render` — pass straight to `LLMClient.complete`.

    - `cached_preamble`: stable text destined for `cache_control=ephemeral`.
    - `user_prompt`: dynamic per-call text.
    - `version`: this prompt's YAML version. Used as the cache-key suffix so
      bumping it in YAML alone invalidates the disk response cache.
    """
    cached_preamble: str
    user_prompt: str
    version: str


class PromptRegistry:
    """Loads prompt YAMLs from a directory at construction time and renders
    them on demand. Thread-safe (Jinja Environment + a small lock around the
    template cache).
    """

    _CACHED_KEY = "cached_preamble"
    _USER_KEY = "user_prompt"
    _VERSION_KEY = "version"

    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        self._lock = RLock()
        self._raw: dict[str, dict] = {}
        self._templates: dict[str, tuple[Template, Template]] = {}
        self._env = Environment(
            undefined=StrictUndefined,        # missing context vars raise — fail loud
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,                 # prompts are plain text, not HTML
        )
        self._discover()

    def _discover(self) -> None:
        """Eagerly load every `*.yaml` in the prompts dir. Skips files starting
        with `_` (treat as partials / scratch)."""
        if not self._prompts_dir.exists():
            logger.warning("prompts_dir_missing", extra={"path": str(self._prompts_dir)})
            return
        for path in sorted(self._prompts_dir.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            name = path.stem
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:  # noqa: BLE001
                logger.error("prompt_yaml_load_failed",
                             extra={"path": str(path), "error": str(e)})
                continue
            self._validate(name, data)
            self._raw[name] = data
            self._templates[name] = (
                self._env.from_string(data[self._CACHED_KEY]),
                self._env.from_string(data[self._USER_KEY]),
            )
            logger.info("prompt_loaded",
                        extra={"prompt_name": name,
                               "version": data.get(self._VERSION_KEY)})

    def _validate(self, name: str, data: dict) -> None:
        for key in (self._CACHED_KEY, self._USER_KEY, self._VERSION_KEY):
            if key not in data:
                raise ValueError(f"Prompt '{name}' missing required key: '{key}'")
        if not isinstance(data[self._VERSION_KEY], str):
            raise ValueError(f"Prompt '{name}' version must be a string, "
                             f"got {type(data[self._VERSION_KEY]).__name__}")

    def render(
        self,
        name: str,
        *,
        cached_context: dict,
        user_context: dict,
    ) -> RenderedPrompt:
        """Render a prompt with separate contexts for the cached and dynamic parts.

        Keeping the two contexts separate enforces the cache contract at the
        call site — accidentally putting per-survey data into `cached_context`
        is a code change, not a silent cache miss.
        """
        if name not in self._templates:
            raise KeyError(f"Unknown prompt: '{name}'. "
                           f"Available: {sorted(self._templates)}")
        cached_tpl, user_tpl = self._templates[name]
        version = self._raw[name][self._VERSION_KEY]
        try:
            cached_preamble = cached_tpl.render(**cached_context)
            user_prompt = user_tpl.render(**user_context)
        except Exception as e:  # noqa: BLE001
            logger.error("prompt_render_failed",
                         extra={"prompt_name": name, "error": str(e)})
            raise
        return RenderedPrompt(
            cached_preamble=cached_preamble,
            user_prompt=user_prompt,
            version=version,
        )

    def get_version(self, name: str) -> str:
        return self._raw[name][self._VERSION_KEY]

    def names(self) -> list[str]:
        return sorted(self._templates)

    def reload(self) -> None:
        """Re-read all prompt files from disk. Useful in dev / tests."""
        with self._lock:
            self._raw.clear()
            self._templates.clear()
            self._discover()


# Module-level singleton — first call wins. Tests can swap via `set_registry`.
_default_registry: PromptRegistry | None = None
_default_lock = RLock()


def _default_prompts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "prompts"


def get_registry() -> PromptRegistry:
    """Return the process-wide prompt registry, building it on first call."""
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = PromptRegistry(_default_prompts_dir())
    return _default_registry


def set_registry(registry: PromptRegistry | None) -> None:
    """Replace the process-wide registry. Pass None to force rebuild on next get."""
    global _default_registry
    with _default_lock:
        _default_registry = registry
