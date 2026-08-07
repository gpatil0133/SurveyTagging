"""JWT auth dependency — PARKED, off by default.

Auth is wired at product integration. Until then `settings.auth_enabled` is
False and `require_auth` is a pass-through, so the API is open in dev.

To enforce auth, set `SURVEY_TAGGER_AUTH_ENABLED=true` and attach the dependency
either globally:

    from auth import require_auth
    app = FastAPI(dependencies=[Depends(require_auth)])

or per-route:

    @app.post("/api/tenants/{t}/surveys/{s}/tag")
    async def tag_survey(t: int, s: int, principal=Depends(require_auth)): ...

Verification: RS256 against keys/public.pem. When `dev_auth_bypass=True` the
Bearer value is treated as a literal numeric corp_no (NO signature check) — dev
environments only; MUST be false in qa/beta/live.
"""

from __future__ import annotations

import logging

from fastapi import Header, HTTPException

from settings import Settings

logger = logging.getLogger("survey_tagging.auth")

_settings = Settings()
_public_key_cache: str | None = None


def _public_key() -> str:
    global _public_key_cache
    if _public_key_cache is None:
        _public_key_cache = _settings.jwt_public_key_path.read_text(encoding="utf-8")
    return _public_key_cache


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header (expected Bearer token)")
    return authorization.split(" ", 1)[1].strip()


async def require_auth(authorization: str | None = Header(default=None)) -> dict | None:
    """FastAPI dependency. Returns the principal (claims dict) or None.

    Pass-through when `auth_enabled` is False. With `dev_auth_bypass` the Bearer
    value is a literal corp_no. Otherwise a RS256 JWT is verified.
    """
    if not _settings.auth_enabled:
        return None

    token = _bearer_token(authorization)

    if _settings.dev_auth_bypass:
        if not token.isdigit():
            raise HTTPException(401, "dev_auth_bypass: Bearer value must be a numeric corp_no")
        return {"corp_no": int(token), "source": "dev_bypass"}

    try:
        import jwt  # PyJWT
        claims = jwt.decode(token, _public_key(), algorithms=[_settings.jwt_algorithm])
        return claims
    except Exception as e:  # noqa: BLE001
        logger.warning("jwt_verification_failed", extra={"error": str(e)})
        raise HTTPException(401, f"Invalid token: {e}")
