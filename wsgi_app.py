r"""WSGI entry point — IIS FastCGI fallback (a2wsgi bridge).

NOT the supported deployment. The supported one is web.config's ARR rewrite in
front of `python run.py` running as a service (see docs/deployment-iis.md);
this exists for hosts where Application Request Routing cannot be installed,
which is the one situation that rule cannot be made to work.

What you give up by using it: the app is started and stopped by the IIS worker
process, so the auto-retag scheduler only runs while some request happens to be
keeping the process alive. Not a correctness problem, but it is why ARR is
preferred. (The other cost used to be the embedding warm-up in run.py's
lifespan, re-paid after every idle-timeout recycle; V9 removed the model, so a
recycle is now cheap.)

Dependency, declared as an extra because it is only needed on this path:

    .venv\Scripts\pip install -e .[iis]

IIS FastCGI handler configuration:

    Full Path:   C:\path\to\Research.SurveyTagging\.venv\Scripts\python.exe
    Arguments:   -u C:\path\to\Research.SurveyTagging\wsgi_app.py
    Environment: SURVEY_TAGGER_PATH_PREFIX = /apisurveytagging   (if mounted
                 under a virtual path; omit at the site root)

Unlike the ARR path, FastCGI does not strip the virtual path before the app
sees it — IIS hands the whole URL over and expects the application to split it
into SCRIPT_NAME + PATH_INFO itself. That split is what `app()` below does, and
it is the only reason this file contains logic rather than two imports.
"""
from __future__ import annotations

import os
import sys

# The package modules (settings, run, service, …) import each other by bare
# name, so this directory has to be importable before anything below runs.
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from a2wsgi import ASGIMiddleware  # noqa: E402

# run.py loads .env itself at import, so SURVEY_TAGGER_* set in the file (not
# just in the FastCGI environment) is honoured here exactly as under uvicorn.
# Importing it does not start a server: the uvicorn.run call is under a
# `if __name__ == "__main__"` guard.
# `_strip_path_prefix` is shared with run.py's PathPrefixMiddleware, which does
# the same job one layer in (and is what makes `python run.py` work under both
# URL shapes). One implementation, so the two paths cannot disagree about what
# counts as the prefix. Stripping here as well as there is harmless: the second
# pass sees a path that no longer carries it and returns it untouched.
from run import _settings, _strip_path_prefix, app as fastapi_app  # noqa: E402

_prefix = _settings.path_prefix  # normalized to "" or "/segment" by Settings

_asgi_wsgi_app = ASGIMiddleware(fastapi_app)


def app(environ, start_response):
    """WSGI entrypoint — strips the sub-app prefix before forwarding to FastAPI.

    The prefix is removed from PATH_INFO and SCRIPT_NAME is deliberately left
    empty, rather than the two being split the way WSGI convention suggests.

    SCRIPT_NAME becomes the ASGI `root_path`, and Starlette will only strip a
    root_path off a path that still carries it. Whether a2wsgi rebuilds `path`
    as SCRIPT_NAME + PATH_INFO (spec-correct) or passes PATH_INFO alone varies
    by version, and under the second mapping a populated root_path makes every
    Mount — i.e. all of /static — serve 404s while top-level routes keep
    working, which is a miserable thing to debug. Stripping here is correct
    under either mapping.

    Nothing is lost by it: the app learns its public base from
    `path_prefix`, not from the request, so this leaves the FastCGI path seeing
    exactly what the ARR path sees. See run.py's note above `app = FastAPI(...)`.
    """
    path = environ.get("PATH_INFO", "/") or "/"
    stripped = _strip_path_prefix(path, _prefix)
    if stripped != path:
        # Copied, not mutated: the server owns `environ` and reuses it.
        environ = environ.copy()
        environ["PATH_INFO"] = stripped
    return _asgi_wsgi_app(environ, start_response)


# IIS FastCGI expects a callable named "application".
application = app
