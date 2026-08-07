"""Filesystem helpers shared by the writers.

Everything the pipeline produces now lands on a Windows network share
(`share_root`), where a plain `open(path, "w")` is genuinely risky: a reader can
observe a torn file mid-write, and the API surfaces that as a 500. Writes go
through `write_json_atomic` instead — temp file in the same directory, then
rename, which SMB2 performs atomically within a share.

The temp files are dot-prefixed on purpose: the change detector skips dotfiles
so a leftover `.tagged_output.json.<rand>.tmp` from a failed rename cannot
poison a survey's input hash.

The staging + rename lives in `sharefs.write_atomic_text` because it has to work
against a UNC root as well as a local one, and `tempfile` only knows the local
filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sharefs


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialize `payload` to `path` via a same-directory temp file + rename."""
    text = json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
    sharefs.write_atomic_text(Path(path), text, encoding="utf-8")
