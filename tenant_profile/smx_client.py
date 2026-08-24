"""Client for the SoGo Research API's `/AIAccountProfile` surface (apismx).

This is the read side of `profile_source="smx"`: instead of paying Parallel.ai
to research a tenant, we read the profile the Research API already generated.
It runs the same three agents we do — profileType 1/2/3 map to our org/cx/ex —
so the artifacts written to the share are the same either way.

Three things about the wire format that the swagger does not tell you:

  1. Responses are **encrypted**. The body is `{"payload": "<base64>"}`; the
     plaintext is recovered by POSTing it to apipmx `/dcdata`. Both base URLs
     therefore have to track the same host, which is why they are derived
     together from `sogo_host` (see settings._derive_sogo_urls_from_host).
  2. Request bodies are **encrypted too** — the direction we missed for a
     while. Every write endpoint on the platform (`/Generate` here, the
     apicx `insertupdate` family elsewhere) Base-64 decodes its body
     server-side before parsing it, so a plain JSON body is answered with
     HTTP 500 "input is not a valid Base-64 string". The body goes through
     apipmx `/ecdata` first and is posted as the RAW ciphertext with
     `Content-Type: text/plain` — not JSON-wrapped, not JSON-quoted; both of
     those reproduce the same 500. See `_encrypt` and `_request`.
  3. The decrypted keys are **PascalCase** (`Data`, `Items`, `CorporateNo`)
     while the swagger advertises camelCase. Every lookup here goes through
     `_ci` so neither spelling can break us.

GET routes (`/List`, `/Details`, `/PackageTypes`) carry their arguments in the
query string, and the platform does not encrypt query strings — only bodies.
So encryption here keys off "does this call have a body", not off the method.

Every route needs a Bearer JWT from the same issuer as our own `auth.py`, so a
request-scoped caller can forward its token; `settings.smx_token` is the
fallback for headless paths (CLI backfill, scheduler).

Every step of every exchange goes through `smx_trace.SmxTrace` — see that module
for what is recorded and why the levels are the way they are. The short version:
failures are logged in full unconditionally, and the successful traffic is one
env flag away.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from tenant_profile.smx_trace import DISABLED, SmxTrace, compact

logger = logging.getLogger(__name__)

# How much of a failing response body goes into the exception message — which
# ends up in the HTTP error the UI shows. The full body is always in the log;
# this is just the part a human sees without opening app.log.
_ERROR_EXCERPT = 500

# profileType -> our agent name. Confirmed against qauc corp 7594:
#   1 "Account" -> org, 2 "CX" -> cx, 3 "EX" -> ex.
PROFILE_TYPE_TO_AGENT: dict[int, str] = {1: "org", 2: "cx", 3: "ex"}


class SmxClientError(Exception):
    """Transport, auth, encrypt, decrypt, or envelope failure talking to apismx."""


# `/ecdata` answers with the ciphertext as a bare JSON-quoted string on every
# deploy we have seen. A couple of platform services wrap it in an object
# instead, and which key they use is not consistent, so probe the known ones
# before giving up. Order matters: `encrypted` is the most specific and wins
# over a deploy that also mirrors the ciphertext into a generic `data`.
_ENCRYPT_RESPONSE_KEYS = ("encrypted", "payload", "data", "result")


@dataclass(frozen=True)
class ProfileRow:
    """One `/AIAccountProfile/Details` row — the SMX analogue of our envelope."""

    profile_type: int
    profile_type_name: str
    agent: str | None            # org/cx/ex, or None when profile_type is unknown
    is_success: bool
    api_status_message: str
    payload: dict[str, Any] | None   # parsed profileResponse
    created_at: str


@dataclass(frozen=True)
class AccountRow:
    """One `/AIAccountProfile/List` item."""

    corporate_no: int
    corporate_id: str
    website_url: str
    effective_url: str
    email_address: str
    package_name: str
    account_status: str
    status: str                  # "Completed" | "Not Generated" | "Failed" | ...
    can_generate: bool
    error_message: str

    @property
    def has_profile(self) -> bool:
        return self.status.strip().lower() in {"completed", "partial"}

    @property
    def website_is_derived(self) -> bool:
        """True when `EffectiveUrl` was inferred from the email domain.

        Most accounts carry no `WebsiteUrl`, and the service falls back to the
        address' domain — which is frequently a reseller or SoGo-internal domain
        shared by many unrelated tenants. Profiles generated from such a URL
        describe the wrong company, so callers should treat this as unverified
        rather than as a curated website.
        """
        if self.website_url.strip():
            return False
        domain = self.email_address.partition("@")[2].strip().lower()
        return bool(domain) and domain in self.effective_url.strip().lower()


def _ci(obj: Any, *names: str) -> Any:
    """Case-insensitive key lookup — the wire is PascalCase, swagger is camelCase."""
    if not isinstance(obj, dict):
        return None
    lowered = {k.lower(): v for k, v in obj.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _read_ciphertext(resp: httpx.Response) -> str:
    """Pull the ciphertext out of an `/ecdata` reply. Empty string if there isn't one.

    The normal case is a bare JSON-quoted string, which `resp.json()` unquotes
    for us. The rest of this handles deploys that answer with a wrapper object,
    or with the raw ciphertext as unparseable text/plain.
    """
    try:
        parsed: Any = resp.json()
    except ValueError:
        # Not JSON at all — the body IS the ciphertext.
        return resp.text.strip()
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in _ENCRYPT_RESPONSE_KEYS:
            value = _ci(parsed, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _row_summary(row: "ProfileRow") -> dict[str, Any]:
    """What a `/Details` row is, without its (already-persisted) contents."""
    return {
        "profile_type": row.profile_type,
        "profile_type_name": row.profile_type_name,
        "agent": row.agent,
        "is_success": row.is_success,
        "api_status_message": row.api_status_message,
        "payload_parsed": row.payload is not None,
        "payload_keys": sorted(row.payload)[:20] if row.payload else [],
        "created_at": row.created_at,
    }


def parse_profile_response(raw: Any) -> dict[str, Any] | None:
    """`profileResponse` is typed `string`; recover the object it holds.

    Returns None when the content cannot be read as a JSON object — callers
    treat that as a missing artifact rather than raising, so one bad agent row
    cannot cost us the other two.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    # A double-encoded payload decodes to a string that is itself JSON.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


class SmxClient:
    """Blocking client for `/AIAccountProfile`. Build one per request or per run."""

    def __init__(
        self,
        *,
        base_url: str,
        pmx_base_url: str,
        token: str,
        timeout: float = 60.0,
        verify: bool | str = True,
        trace: SmxTrace | None = None,
        encrypt_requests: bool = True,
    ) -> None:
        if not token:
            raise SmxClientError(
                "No apismx bearer token. Forward the caller's JWT, or set "
                "SURVEY_TAGGER_SMX_TOKEN for headless runs."
            )
        self._base = base_url.rstrip("/")
        self._pmx = pmx_base_url.rstrip("/")
        self._trace = trace or DISABLED
        # Off only as an escape hatch for a deploy that has not turned request
        # encryption on yet; see settings.smx_encrypt_requests.
        self._encrypt_requests = bool(encrypt_requests)
        self._client = httpx.Client(
            timeout=timeout,
            verify=verify,
            headers={
                "Authorization": f"Bearer {token.removeprefix('Bearer ').strip()}",
                "Accept": "application/json",
            },
        )

    def __enter__(self) -> "SmxClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ---------- transport ----------

    def _encrypt(self, payload: Any, exchange: str = "") -> str:
        """Turn a request body into the ciphertext apismx expects, via `/ecdata`.

        The mirror of `_decrypt`, and traced the same way for the same reason: a
        `/Generate` that 500s because the body never got encrypted looks exactly
        like a `/Generate` that 500s server-side, and only the separate
        `smx_encrypt_*` records tell them apart.

        Two details are load-bearing and were learned the hard way on the
        platform's other services:

          * The plaintext must be **compact** JSON (`separators=(",", ":")`).
            The endpoint is whitespace-sensitive in places, so a pretty-printed
            body can round-trip into something the target route rejects.
          * The reply is the ciphertext as a bare JSON-quoted string, e.g.
            `"L5apItJtPULTQoy3rBF24tlUoFqUidiAMcPEkWtoh5I="`. The outer quotes
            are JSON, not part of the ciphertext, and forwarding them is one of
            the two ways to earn a Base-64 decode error downstream.
        """
        # ensure_ascii left at its default on purpose: the plaintext goes out as
        # pure ASCII with \uXXXX escapes, which survives the Base-64 round trip
        # and the .NET side's decoding no matter what a tenant name contains.
        plaintext = (payload if isinstance(payload, str)
                     else json.dumps(payload, separators=(",", ":")))
        url = f"{self._pmx}/ecdata"
        started = time.perf_counter()
        self._trace.record("smx_encrypt_request", exchange=exchange, url=url,
                           plaintext=plaintext)
        try:
            resp = self._client.post(
                url,
                json={"ecValue": plaintext},
                # `Accept: text/plain` is what the platform's other services send
                # here, and it matches what comes back — a bare quoted string
                # rather than a JSON document. Overrides the client-wide
                # `Accept: application/json` for this call only.
                headers={"Content-Type": "application/json", "Accept": "text/plain"},
            )
        except httpx.HTTPError as e:
            self._trace.record("smx_encrypt_transport_error", level=logging.ERROR,
                               exchange=exchange, url=url, error=str(e),
                               duration_ms=_ms_since(started))
            raise SmxClientError(f"/ecdata request failed: {e}") from e
        duration_ms = _ms_since(started)
        if resp.status_code != 200:
            self._trace.record("smx_encrypt_http_error", level=logging.ERROR,
                               exchange=exchange, url=url,
                               status_code=resp.status_code, duration_ms=duration_ms,
                               response_body=resp.text)
            raise SmxClientError(
                f"/ecdata returned HTTP {resp.status_code}: {resp.text[:_ERROR_EXCERPT]}"
            )

        ciphertext = _read_ciphertext(resp)
        if not ciphertext:
            self._trace.record("smx_encrypt_unreadable", level=logging.ERROR,
                               exchange=exchange, url=url, duration_ms=duration_ms,
                               response_body=resp.text)
            raise SmxClientError(
                f"/ecdata returned no usable ciphertext: {resp.text[:_ERROR_EXCERPT]}"
            )
        # Length only. The ciphertext is opaque and the plaintext is already on
        # the `smx_encrypt_request` line, so recording it again is pure noise.
        self._trace.record("smx_encrypted", exchange=exchange, duration_ms=duration_ms,
                           plaintext_chars=len(plaintext),
                           ciphertext_chars=len(ciphertext))
        return ciphertext

    def _decrypt(self, blob: str, exchange: str = "") -> Any:
        """Round-trip an encrypted payload through apipmx `/dcdata`.

        Traced separately from the call that produced the blob: a `/Details` that
        returns 200 and then fails to decrypt is a different problem (mismatched
        apismx/apipmx hosts) from one that never answered, and the one-line
        `smx_decrypted` record with the plaintext is what makes the difference
        visible.
        """
        url = f"{self._pmx}/dcdata"
        started = time.perf_counter()
        self._trace.record("smx_decrypt_request", exchange=exchange, url=url,
                           blob_chars=len(blob))
        try:
            resp = self._client.post(url, json={"decValue": blob})
        except httpx.HTTPError as e:
            self._trace.record("smx_decrypt_transport_error", level=logging.ERROR,
                               exchange=exchange, url=url, error=str(e),
                               duration_ms=_ms_since(started))
            raise SmxClientError(f"/dcdata request failed: {e}") from e
        duration_ms = _ms_since(started)
        if resp.status_code != 200:
            self._trace.record("smx_decrypt_http_error", level=logging.ERROR,
                               exchange=exchange, url=url,
                               status_code=resp.status_code, duration_ms=duration_ms,
                               response_body=resp.text)
            raise SmxClientError(
                f"/dcdata returned HTTP {resp.status_code}: {resp.text[:_ERROR_EXCERPT]}"
            )
        try:
            inner = resp.json()
        except ValueError:
            inner = resp.text
        # /dcdata answers with a bare JSON string holding the real document.
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError as e:
                self._trace.record("smx_decrypt_non_json", level=logging.ERROR,
                                   exchange=exchange, url=url, error=str(e),
                                   response_body=inner)
                raise SmxClientError(f"/dcdata returned non-JSON plaintext: {e}") from e
        self._trace.record("smx_decrypted", exchange=exchange, duration_ms=duration_ms,
                           plaintext=compact(inner))
        return inner

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}{path}"
        exchange = self._trace.new_exchange()
        params = kwargs.get("params")
        # Taken out of kwargs: an encrypted call sends `content`, not `json`, and
        # httpx rejects being handed both.
        request_body = kwargs.pop("json", None)
        encrypt = request_body is not None and self._encrypt_requests
        self._trace.record("smx_request", exchange=exchange, method=method, url=url,
                           params=params, request_body=request_body,
                           encrypted_request=encrypt)
        if encrypt:
            # Raw ciphertext as text/plain. Wrapping it in `{"payload": ...}` or
            # leaving the JSON quotes on both come back as HTTP 500 "input is
            # not a valid Base-64 string" — the controller Base-64 decodes the
            # body before it parses anything.
            kwargs["content"] = self._encrypt(request_body, exchange)
            kwargs["headers"] = {**(kwargs.get("headers") or {}),
                                 "Content-Type": "text/plain"}
        elif request_body is not None:
            kwargs["json"] = request_body
        # Started after the encrypt round-trip so this duration is the apismx
        # call alone; `/ecdata`'s own timing is on `smx_encrypted`.
        started = time.perf_counter()
        try:
            resp = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            self._trace.record("smx_transport_error", level=logging.ERROR,
                               exchange=exchange, method=method, url=url,
                               error=str(e), duration_ms=_ms_since(started))
            raise SmxClientError(f"{method} {path} failed: {e}") from e
        duration_ms = _ms_since(started)

        if resp.status_code >= 400:
            # ERROR, and the WHOLE body — the exception carries an excerpt to the
            # caller, but the server's own explanation of a 500 exists nowhere
            # else. This is the one record that must not need a flag flipped.
            self._trace.record("smx_http_error", level=logging.ERROR,
                               exchange=exchange, method=method, url=url,
                               params=params, request_body=request_body,
                               status_code=resp.status_code, duration_ms=duration_ms,
                               response_body=resp.text)
            if resp.status_code == 401:
                raise SmxClientError(
                    f"{method} {path} returned 401 — the apismx token is missing, "
                    "expired, or issued for a different environment."
                )
            raise SmxClientError(
                f"{method} {path} returned HTTP {resp.status_code}: "
                f"{resp.text[:_ERROR_EXCERPT]}"
            )
        try:
            body = resp.json()
        except ValueError as e:
            self._trace.record("smx_non_json_response", level=logging.ERROR,
                               exchange=exchange, method=method, url=url,
                               status_code=resp.status_code, duration_ms=duration_ms,
                               response_body=resp.text)
            raise SmxClientError(f"{method} {path} returned non-JSON body: {e}") from e

        blob = _ci(body, "payload")
        encrypted = isinstance(blob, str) and bool(blob)
        # An encrypted body is a base64 blob: recording it would be pure noise,
        # so the size goes here and the content comes from `smx_decrypted`.
        self._trace.record("smx_response", exchange=exchange, method=method, url=url,
                           status_code=resp.status_code, duration_ms=duration_ms,
                           encrypted=encrypted,
                           payload_chars=len(blob) if encrypted else None,
                           response_body=None if encrypted else compact(body))
        if encrypted:
            body = self._decrypt(blob, exchange)

        message = _ci(body, "message") or {}
        data = _ci(body, "data")
        self._trace.record("smx_envelope", exchange=exchange, url=url,
                           envelope_status=_ci(message, "status"),
                           user_message=_as_str(_ci(message, "userMessage")) or None,
                           exception_message=_as_str(_ci(message, "exceptionMessage")) or None,
                           data_type=type(data).__name__,
                           data_len=len(data) if isinstance(data, (list, dict, str)) else None)
        return self._unwrap(body, f"{method} {path}")

    @staticmethod
    def _unwrap(body: Any, label: str) -> Any:
        """Pull `Data` out of the `{Data, Message}` envelope, raising on failure."""
        message = _ci(body, "message") or {}
        data = _ci(body, "data")
        if _ci(message, "status") is False:
            detail = (_as_str(_ci(message, "userMessage"))
                      or _as_str(_ci(message, "exceptionMessage"))
                      or "no detail")
            raise SmxClientError(f"{label} failed: {detail}")
        return data

    # ---------- endpoints ----------

    def list_accounts(
        self,
        *,
        page_number: int = 1,
        page_size: int = 50,
        package_name: str = "",
        search: str = "",
        status_filter: int | None = None,
        sort_column: str = "corporateNo",
        sort_order: str = "asc",
    ) -> tuple[list[AccountRow], dict[str, Any]]:
        """One page of `/AIAccountProfile/List`.

        Returns (rows, page_meta) where page_meta carries TotalRecords,
        HasNextPage and the Completed/Failed/Partial/NotGenerated roll-ups.
        """
        params: dict[str, Any] = {
            "search": search,
            "pageNumber": page_number,
            "pageSize": page_size,
            "sortColumn": sort_column,
            "sortOrder": sort_order,
        }
        if package_name:
            params["packageName"] = package_name
        if status_filter is not None:
            params["statusFilter"] = status_filter

        data = self._request("GET", "/AIAccountProfile/List", params=params) or {}
        items = _ci(data, "items") or []
        rows = [
            AccountRow(
                corporate_no=int(_ci(it, "corporateNo") or 0),
                corporate_id=_as_str(_ci(it, "corporateId")),
                website_url=_as_str(_ci(it, "websiteUrl")),
                effective_url=_as_str(_ci(it, "effectiveUrl")),
                email_address=_as_str(_ci(it, "emailAddress")),
                package_name=_as_str(_ci(it, "packageName")),
                account_status=_as_str(_ci(it, "accountStatus")),
                status=_as_str(_ci(it, "status")),
                can_generate=bool(_ci(it, "canGenerate")),
                error_message=_as_str(_ci(it, "errorMessage")),
            )
            for it in items
            if isinstance(it, dict)
        ]
        meta = {k: v for k, v in data.items() if k.lower() != "items"} if isinstance(data, dict) else {}
        return rows, meta

    def iter_accounts(
        self, *, package_name: str = "", page_size: int = 50, max_pages: int = 200,
    ):
        """Walk every page of `/AIAccountProfile/List`.

        `max_pages` is a runaway guard: the roll-up counters are repeated on
        every row, so a server-side paging bug would otherwise loop forever.
        """
        for page in range(1, max_pages + 1):
            rows, meta = self.list_accounts(
                page_number=page, page_size=page_size, package_name=package_name,
            )
            yield from rows
            if not rows or not _ci(meta, "hasNextPage"):
                return
        logger.warning("smx_list_pagination_capped", extra={"max_pages": max_pages})

    def get_details(self, corp_no: int) -> list[ProfileRow]:
        """All generated profile rows for one tenant. Empty list = never generated."""
        data = self._request("GET", "/AIAccountProfile/Details",
                             params={"corpNo": corp_no}) or []
        if not isinstance(data, list):
            raise SmxClientError(
                f"Details for corp {corp_no} returned {type(data).__name__}, expected a list"
            )
        rows: list[ProfileRow] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ptype_raw = _ci(item, "profileType")
            try:
                ptype = int(ptype_raw)
            except (TypeError, ValueError):
                logger.warning("smx_unreadable_profile_type",
                               extra={"corp_no": corp_no, "value": repr(ptype_raw)})
                continue
            raw_response = _ci(item, "profileResponse")
            payload = parse_profile_response(raw_response)
            if payload is None:
                # The row exists but carries nothing we can read. Keep the raw
                # string: "generated but empty" and "generated but malformed"
                # look identical downstream, and only this tells them apart.
                self._trace.record("smx_unreadable_profile_response",
                                   corp_no=corp_no, profile_type=ptype,
                                   raw_type=type(raw_response).__name__,
                                   raw=_as_str(raw_response))
            rows.append(ProfileRow(
                profile_type=ptype,
                profile_type_name=_as_str(_ci(item, "profileTypeName")),
                agent=PROFILE_TYPE_TO_AGENT.get(ptype),
                is_success=bool(_ci(item, "isSuccess")),
                api_status_message=_as_str(_ci(item, "apiStatusMessage")),
                payload=payload,
                created_at=_as_str(_ci(item, "createdate", "createDate", "created_at")),
            ))

        # The shape of what came back, not its content — the content is already
        # written to the share as an artifact. This is the record that answers
        # "does this tenant have a profile, and is it usable?".
        self._trace.record("smx_details_rows", corp_no=corp_no, row_count=len(rows),
                           rows=[_row_summary(r) for r in rows])

        unknown = [r.profile_type for r in rows if r.agent is None]
        if unknown:
            # Not fatal: a new profile type is additive, and the three we map
            # still produce complete artifacts.
            logger.warning("smx_unknown_profile_types",
                           extra={"corp_no": corp_no, "profile_types": unknown})
        return rows

    def generate(self, corp_nos: list[int]) -> dict[str, Any]:
        """Trigger generation. Write operation — callers gate this on user intent.

        The only route here with a request body, and therefore the only one
        `_request` encrypts. Pass the body as a plain dict: encryption is
        `_request`'s job, so nothing above this line has to know about `/ecdata`.
        """
        if not corp_nos:
            raise SmxClientError("generate() needs at least one corp_no")
        body = {"corpNos": [int(c) for c in corp_nos]}
        summary = self._request("POST", "/AIAccountProfile/Generate", json=body) or {}
        # inserted/skipped/failed. "Accepted" and "actually queued" are not the
        # same thing, and this is the only place the difference shows.
        self._trace.record("smx_generate_accepted", corp_nos=body["corpNos"],
                           summary=summary)
        return summary

    def package_types(self) -> list[str]:
        data = self._request("GET", "/AIAccountProfile/PackageTypes") or []
        return [str(x) for x in data] if isinstance(data, list) else []
