"""Filesystem facade that speaks both local paths and SMB/UNC shares.

The share root can be given either as a local path (a mount point, or a local
dev folder) or as a UNC path pointing straight at the image server:

    SURVEY_TAGGER_SHARE_ROOT=/mnt/aichatbot/AIChatbot            -> local
    SURVEY_TAGGER_SHARE_ROOT=//172.16.1.105/sg-int-uc-img/...    -> SMB

In the UNC case nothing is mounted: the process talks SMB2/3 itself via
`smbprotocol`, authenticating with SURVEY_TAGGER_IMAGE_USER / _IMAGE_PASS. That
removes the root-owned CIFS mount from the deployment, at the cost of every
file operation having to route through here rather than through pathlib.

Why a function facade instead of a Path subclass
------------------------------------------------
`pathlib` already handles UNC *path algebra* correctly on POSIX — it preserves
exactly two leading slashes as a distinct root, so `Path("//host/share") / "x"`,
`.parent`, `.name`, `.relative_to` and `.with_suffix` all behave. What it cannot
do off Windows is *open* such a path. So `Path` stays the path-carrying type
everywhere in the codebase and only the I/O verbs move here.

Every function dispatches on the path itself, so a single call site serves both
deployments and local development keeps working untouched. Paths that never live
on the share (`cache_dir`, `config/`, the JWT key, static assets) deliberately
still use plain pathlib at their call sites — routing them here would work, but
it would suggest they might be remote when they never are.

Threading: smbprotocol keeps a locked connection pool keyed by server, so the
scheduler's threads and `max_concurrent_surveys` workers share one connection
safely.
"""

from __future__ import annotations

import fnmatch
import logging
import uuid
from pathlib import Path, PurePosixPath
from typing import IO, Any

logger = logging.getLogger("survey_tagging.sharefs")

# Set by configure(); read lazily so importing this module never requires
# smbprotocol to be installed for a purely-local deployment.
_username: str | None = None
_password: str | None = None
_registered: set[str] = set()


# ---------- UNC detection / normalization ----------


def is_unc(path: Any) -> bool:
    """True for `//server/share/...` (or the backslash spelling).

    Exactly two leading separators. POSIX reserves `//x` as implementation-
    defined and pathlib preserves it; three or more collapse to one and are an
    ordinary absolute path.
    """
    s = str(path)
    if len(s) < 3:
        return False
    if s[0] not in "/\\" or s[1] not in "/\\":
        return False
    return s[2] not in "/\\"


def normalize(path: Any) -> Path:
    """Canonicalize a share root to the `//server/share/...` spelling.

    A backslash UNC path is a single meaningless component to PosixPath, so it
    has to be rewritten before anything can join onto it. Forward slashes are
    understood by both pathlib flavours and by smbprotocol (which runs the value
    through `ntpath.normpath`), so this spelling is the one that works
    everywhere.
    """
    s = str(path)
    if is_unc(s):
        return Path("//" + s[2:].replace("\\", "/").lstrip("/"))
    return Path(s)


def unc_parts(path: Any) -> tuple[str, ...]:
    """Components of a normalized UNC root, server first: `('server', 'share', ...)`.

    `Path.parts` cannot be used for this. On Windows the flavour is
    `WindowsPath`, which folds the whole `//server/share/` anchor into a single
    part, so a valid root yields `('\\\\server\\share\\', 'dir')` — the server is
    not addressable and the component count is 2 short of the POSIX flavour's.
    Reading it with the posix flavour keeps the answer identical on every
    platform.
    """
    posix = PurePosixPath(normalize(path).as_posix())
    # Leading '//' is its own anchor part under the posix flavour; drop it.
    return tuple(p for p in posix.parts if p not in ("/", "//"))


def server_of(path: Any) -> str:
    """Server component of a UNC path (`//server/share/x` -> `server`)."""
    return unc_parts(path)[0]


def _s(path: Any) -> str:
    """Path as a string for smbclient, which normalizes separators itself."""
    return str(path)


def _p(raw: str) -> Path:
    """A path smbclient handed back (backslash spelling) as a `//` Path."""
    return normalize(raw) if is_unc(raw) else Path(raw)


# ---------- credentials / session ----------


def configure(username: str | None, password: str | None) -> None:
    """Record share credentials. Call once at startup, before any share I/O.

    Credentials are applied as smbclient's process-wide default, so paths on any
    server route through them. `username` may carry a domain as `DOMAIN\\user`
    or `user@domain`.
    """
    global _username, _password
    _username = username or None
    _password = password or None
    if not _username:
        return
    import smbclient

    smbclient.ClientConfig(username=_username, password=_password)
    logger.info("sharefs_credentials_configured", extra={"user": _username})


def connect(path: Any) -> None:
    """Open + cache a session to the server behind `path`, failing loudly.

    Called at startup so a bad password surfaces there rather than as an empty
    tenant list from the first request that touches the share — every discovery
    function swallows OSError by design.
    """
    if not is_unc(path):
        return
    server = server_of(path)
    if server in _registered:
        return
    if not _username:
        raise RuntimeError(
            f"SURVEY_TAGGER_SHARE_ROOT is the UNC path {str(path)!r}, which needs "
            "credentials, but SURVEY_TAGGER_IMAGE_USER is empty. Set "
            "SURVEY_TAGGER_IMAGE_USER and SURVEY_TAGGER_IMAGE_PASS in .env."
        )
    import smbclient

    smbclient.register_session(server, username=_username, password=_password)
    _registered.add(server)
    logger.info("sharefs_session_registered", extra={"server": server})


def probe(path: Any) -> dict:
    """Can we actually reach and list `path` right now? Never raises.

    Returns `{root, reachable, error}`. This is the diagnosis that used to
    happen at startup: because the process no longer connects eagerly (see
    bootstrap.build_context), a bad password or an unreachable server would
    otherwise only ever show up as an empty tenant list — every discovery
    function swallows OSError and cannot tell "no surveys" from "cannot log in".

    `connect()` runs first for a UNC root so a credential failure is reported as
    itself rather than as a missing directory: smbclient surfaces a rejected
    logon from `is_dir` as a bare OSError that reads like a bad path.
    """
    root = normalize(path)
    try:
        if is_unc(root):
            connect(root)
        reachable = is_dir(root)
    except Exception as e:  # noqa: BLE001
        # ValueError (transport refused), OSError (SMB status), RuntimeError
        # (no credentials configured) all land here and all mean the same thing
        # to a caller: the root is not usable, and here is why.
        return {"root": str(root), "reachable": False,
                "error": f"{type(e).__name__}: {e}"}
    return {
        "root": str(root),
        "reachable": reachable,
        "error": None if reachable else "root does not exist or is not a directory",
    }


def reset() -> None:
    """Drop all cached SMB connections (tests, credential rotation)."""
    if _registered:
        import smbclient

        smbclient.reset_connection_cache()
        _registered.clear()


# ---------- queries ----------


def exists(path: Any) -> bool:
    if not is_unc(path):
        return Path(path).exists()
    import smbclient.path

    return smbclient.path.exists(_s(path))


def is_dir(path: Any) -> bool:
    if not is_unc(path):
        return Path(path).is_dir()
    import smbclient.path

    return smbclient.path.isdir(_s(path))


def is_file(path: Any) -> bool:
    if not is_unc(path):
        return Path(path).is_file()
    import smbclient.path

    return smbclient.path.isfile(_s(path))


def stat(path: Any):
    """`os.stat_result`-alike. The SMB result also carries st_size/st_mtime_ns."""
    if not is_unc(path):
        return Path(path).stat()
    import smbclient

    return smbclient.stat(_s(path))


# ---------- listing ----------


def iterdir(path: Any) -> list[Path]:
    """Children of a directory. Eager (a list) so callers cannot hold a cursor
    open across an SMB round trip."""
    if not is_unc(path):
        return list(Path(path).iterdir())
    import smbclient

    base = normalize(path)
    return [base / name for name in smbclient.listdir(_s(path))]


def _has_magic(part: str) -> bool:
    return any(c in part for c in "*?[")


def glob(path: Any, pattern: str) -> list[Path]:
    """`Path.glob` for one or more components (`SurveyData/*/tagged_output.json`).

    Matching is done locally with fnmatch over a plain `*` listing rather than
    by passing the pattern to the server: SMB wildcard semantics have their own
    quirks (notably around `.` and short names) and would not match what the
    local branch does. `**` is not supported — nothing here needs it.
    """
    if not is_unc(path):
        return list(Path(path).glob(pattern))
    import smbclient

    parts = [p for p in str(pattern).replace("\\", "/").split("/") if p]
    if not parts:
        return []
    bases = [normalize(path)]
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        found: list[Path] = []
        for base in bases:
            if _has_magic(part):
                try:
                    entries = list(smbclient.scandir(_s(base)))
                except OSError:
                    continue
                for entry in entries:
                    if entry.name in (".", "..") or not fnmatch.fnmatch(entry.name, part):
                        continue
                    if last or entry.is_dir():
                        found.append(base / entry.name)
            else:
                cand = base / part
                if (exists(cand) if last else is_dir(cand)):
                    found.append(cand)
        bases = found
        if not bases:
            return []
    return bases


def rglob(path: Any, pattern: str = "*") -> list[Path]:
    """Recursive `Path.rglob`, returning files and directories."""
    if not is_unc(path):
        return list(Path(path).rglob(pattern))
    import smbclient

    out: list[Path] = []
    for root, dirs, files in smbclient.walk(_s(path)):
        base = _p(root)
        for name in list(dirs) + list(files):
            if fnmatch.fnmatch(name, pattern):
                out.append(base / name)
    return out


def rglob_files(path: Any, pattern: str = "*") -> list[Path]:
    """Recursive walk yielding files only.

    Separate from `rglob` because the directory walk already knows which
    entries are files. Filtering afterwards with `is_file()` would cost one
    extra round trip per entry, which the change detector — the only caller —
    pays once per file in a tenant tree on every run.
    """
    if not is_unc(path):
        return [p for p in Path(path).rglob(pattern) if p.is_file()]
    import smbclient

    out: list[Path] = []
    for root, _dirs, files in smbclient.walk(_s(path)):
        base = _p(root)
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                out.append(base / name)
    return out


# ---------- mutation ----------


def mkdir(path: Any, *, parents: bool = True, exist_ok: bool = True) -> None:
    if not is_unc(path):
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)
        return
    import smbclient

    if parents:
        smbclient.makedirs(_s(path), exist_ok=exist_ok)
    else:
        try:
            smbclient.mkdir(_s(path))
        except FileExistsError:
            if not exist_ok:
                raise


def unlink(path: Any, *, missing_ok: bool = False) -> None:
    if not is_unc(path):
        Path(path).unlink(missing_ok=missing_ok)
        return
    import smbclient

    try:
        smbclient.remove(_s(path))
    except FileNotFoundError:
        if not missing_ok:
            raise


def replace(src: Any, dst: Any) -> None:
    """Rename over the destination. Atomic within a share, as it is locally."""
    if not is_unc(src):
        Path(src).replace(dst)
        return
    import smbclient

    smbclient.replace(_s(src), _s(dst))


# ---------- open / read / write ----------


def open_file(
    path: Any,
    mode: str = "r",
    *,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> IO[Any]:
    """`open()` for either backend. Use as a context manager.

    Supports the modes the codebase actually uses, including `x` for the
    exclusive create the canon lock relies on.
    """
    if not is_unc(path):
        if "b" in mode:
            return open(path, mode)
        return open(path, mode, encoding=encoding, errors=errors, newline=newline)
    import smbclient

    if "b" in mode:
        return smbclient.open_file(_s(path), mode=mode)
    return smbclient.open_file(
        _s(path), mode=mode, encoding=encoding, errors=errors, newline=newline
    )


def read_bytes(path: Any) -> bytes:
    if not is_unc(path):
        return Path(path).read_bytes()
    with open_file(path, "rb") as f:
        return f.read()


def read_text(path: Any, *, encoding: str = "utf-8", errors: str | None = None) -> str:
    if not is_unc(path):
        return Path(path).read_text(encoding=encoding, errors=errors)
    with open_file(path, "r", encoding=encoding, errors=errors) as f:
        return f.read()


def write_text(path: Any, data: str, *, encoding: str = "utf-8") -> None:
    with open_file(path, "w", encoding=encoding) as f:
        f.write(data)


def write_atomic_text(path: Any, data: str, *, encoding: str = "utf-8") -> None:
    """Write via a same-directory temp file + rename, so a concurrent reader
    never observes a torn file.

    The temp name is dot-prefixed because the change detector skips dotfiles: a
    leftover temp from a failed rename must not perturb a survey's input hash.
    `tempfile` is not usable here — it only knows the local filesystem.
    """
    path = Path(path)
    mkdir(path.parent)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        write_text(tmp, data, encoding=encoding)
        replace(tmp, path)
    except Exception:
        try:
            unlink(tmp, missing_ok=True)
        except OSError:
            pass
        raise
