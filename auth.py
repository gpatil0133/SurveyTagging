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
import time

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


# --------------------------------------------------------------------------
# Token reading (independent of enforcement)
#
# The browser UI puts a bearer token on every request (static/app.js,
# readToken): the platform's `access_token` from localStorage, or one pasted
# into the Tenant Profile panel, which takes precedence over it. Two things are
# done with it, neither of which waits for `auth_enabled` to be flipped on:
#
#   1. It is forwarded verbatim to apismx and any other SoGo service we call
#      outbound — they share an issuer with us, so the caller's own token is
#      the right credential to travel with the request.
#   2. Its corp_no claim answers "which tenant?" for callers that did not put
#      one in the URL. The URL still wins when it has one: some tenants exist
#      only on the net-share and never had a platform account, so the path is
#      the authoritative source and a token/path mismatch is NOT an error.
# --------------------------------------------------------------------------

# Every spelling of the tenant claim seen across Research.Auth-issued tokens.
# Order matters only in that the first non-zero hit wins.
_CORP_CLAIMS = ("corp_no", "corporate_no", "corp_no_um", "corpNo", "corporateNo", "CorpId")


def bearer_token(authorization: str | None) -> str:
    """The raw JWT from an Authorization header, or "" when there isn't one.

    Non-raising counterpart to `_bearer_token` — used on the open routes, which
    must keep working for callers that send no token at all.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    return authorization.split(" ", 1)[1].strip()


def _decode_verified(token: str) -> tuple[dict | None, bool]:
    """`(claims, verified)` from a token; `(None, False)` if it cannot be read.

    Verifies the RS256 signature when the public key is readable. When
    verification fails and `auth_enabled` is False the claims are still decoded
    unverified — the API is open in that mode, so an unverified corp_no grants
    nothing a plain `?tenant_id=` would not, and refusing to read it would just
    break the dev/embedded flow. Once `auth_enabled` is True, `require_auth`
    rejects the request before anything here is consulted.

    `exp` is checked by hand: JOSE-JWT (what the platform issues with) encodes
    it as a string float, which PyJWT's built-in check rejects outright.
    """
    if not token:
        return None, False
    try:
        import jwt  # PyJWT
    except ImportError:
        return None, False

    claims: dict | None = None
    verified = False
    try:
        claims = jwt.decode(
            token, _public_key(), algorithms=[_settings.jwt_algorithm],
            options={"verify_exp": False, "verify_iat": False},
        )
        verified = True
    except Exception as e:  # noqa: BLE001
        if _settings.auth_enabled:
            return None, False
        logger.debug("jwt_unverified_read", extra={"error": str(e)})
        try:
            claims = jwt.decode(token, options={"verify_signature": False,
                                                "verify_exp": False, "verify_iat": False})
        except Exception:  # noqa: BLE001
            return None, False

    if not isinstance(claims, dict):
        return None, False
    raw_exp = claims.get("exp")
    if raw_exp is not None:
        try:
            if time.time() > float(raw_exp):
                logger.debug("jwt_expired")
                return None, False
        except (TypeError, ValueError):
            pass  # unparseable exp — treat as non-expiring rather than as a reject
    return claims, verified


def corp_no_from_claims(claims: dict | None) -> int | None:
    """Tenant id out of a claims dict, trying every known spelling.

    Zero is skipped: some issuers emit `corp_no="0"` as a sentinel for sub-users
    and carry the real tenant in `corp_no_um` / `CorpId`.
    """
    if not claims:
        return None
    for key in _CORP_CLAIMS:
        raw = claims.get(key)
        if raw is None:
            continue
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if value:
            return value
    return None


def principal(authorization: str | None) -> dict:
    """Who the caller is, as far as the token says. Never raises.

    `{"corp_no": int|None, "subject": str|None, "has_token": bool,
      "verified": bool}` — `verified` is False when the signature could not be
    checked but the claims were read anyway (see `_decode`).
    """
    token = bearer_token(authorization)
    if not token:
        return {"corp_no": None, "subject": None, "has_token": False, "verified": False}
    claims, verified = _decode_verified(token)
    return {
        "corp_no": corp_no_from_claims(claims),
        "subject": (claims or {}).get("sub"),
        "has_token": True,
        "verified": verified,
    }


def resolve_tenant_id(tenant_id: int | None, authorization: str | None) -> int:
    """The tenant this request is about: the URL if it named one, else the token.

    The URL is authoritative and is never cross-checked against the token —
    tenants whose data lives only on the net-share have no platform account for
    a token to agree with. The token is the fallback for embedded callers that
    know who they are but do not put it in the path.
    """
    if tenant_id is not None:
        return tenant_id
    corp_no = corp_no_from_claims(_decode_verified(bearer_token(authorization))[0])
    if corp_no is not None:
        return corp_no
    raise HTTPException(
        400,
        "No tenant in the request. Either use the /api/tenants/{tenant_id}/… "
        "form, pass ?tenant_id=, or send an Authorization: Bearer token whose "
        "claims carry a corp_no.",
    )


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

    # `_decode_verified` is the single decode path: with auth_enabled True it
    # never falls back to unverified claims, and it applies the manual `exp`
    # check that a JOSE-issued string-float expiry needs.
    claims, verified = _decode_verified(token)
    if claims is None or not verified:
        logger.warning("jwt_verification_failed")
        raise HTTPException(401, "Invalid or expired token")
    return claims
