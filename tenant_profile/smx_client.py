"""Client for the SoGo Research API's `/AIAccountProfile` surface (apismx).

This is the read side of `profile_source="smx"`: instead of paying Parallel.ai
to research a tenant, we read the profile the Research API already generated.
It runs the same three agents we do — profileType 1/2/3 map to our org/cx/ex —
so the artifacts written to the share are the same either way.

Two things about the wire format that the swagger does not tell you:

  1. Responses are **encrypted**. The body is `{"payload": "<base64>"}`; the
     plaintext is recovered by POSTing it to apipmx `/dcdata`. Both base URLs
     therefore have to track the same host, which is why they are derived
     together from `sogo_host` (see settings._derive_sogo_urls_from_host).
  2. The decrypted keys are **PascalCase** (`Data`, `Items`, `CorporateNo`)
     while the swagger advertises camelCase. Every lookup here goes through
     `_ci` so neither spelling can break us.

Every route needs a Bearer JWT from the same issuer as our own `auth.py`, so a
request-scoped caller can forward its token; `settings.smx_token` is the
fallback for headless paths (CLI backfill, scheduler).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# profileType -> our agent name. Confirmed against qauc corp 7594:
#   1 "Account" -> org, 2 "CX" -> cx, 3 "EX" -> ex.
PROFILE_TYPE_TO_AGENT: dict[int, str] = {1: "org", 2: "cx", 3: "ex"}


class SmxClientError(Exception):
    """Transport, auth, decrypt, or envelope failure talking to apismx."""


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
    ) -> None:
        if not token:
            raise SmxClientError(
                "No apismx bearer token. Forward the caller's JWT, or set "
                "SURVEY_TAGGER_SMX_TOKEN for headless runs."
            )
        self._base = base_url.rstrip("/")
        self._pmx = pmx_base_url.rstrip("/")
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

    def _decrypt(self, blob: str) -> Any:
        """Round-trip an encrypted payload through apipmx `/dcdata`."""
        try:
            resp = self._client.post(f"{self._pmx}/dcdata", json={"decValue": blob})
        except httpx.HTTPError as e:
            raise SmxClientError(f"/dcdata request failed: {e}") from e
        if resp.status_code != 200:
            raise SmxClientError(
                f"/dcdata returned HTTP {resp.status_code}: {resp.text[:200]}"
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
                raise SmxClientError(f"/dcdata returned non-JSON plaintext: {e}") from e
        return inner

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}{path}"
        try:
            resp = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise SmxClientError(f"{method} {path} failed: {e}") from e

        if resp.status_code == 401:
            raise SmxClientError(
                f"{method} {path} returned 401 — the apismx token is missing, "
                "expired, or issued for a different environment."
            )
        if resp.status_code >= 400:
            raise SmxClientError(
                f"{method} {path} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            body = resp.json()
        except ValueError as e:
            raise SmxClientError(f"{method} {path} returned non-JSON body: {e}") from e

        blob = _ci(body, "payload")
        if isinstance(blob, str) and blob:
            body = self._decrypt(blob)
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
            rows.append(ProfileRow(
                profile_type=ptype,
                profile_type_name=_as_str(_ci(item, "profileTypeName")),
                agent=PROFILE_TYPE_TO_AGENT.get(ptype),
                is_success=bool(_ci(item, "isSuccess")),
                api_status_message=_as_str(_ci(item, "apiStatusMessage")),
                payload=parse_profile_response(_ci(item, "profileResponse")),
                created_at=_as_str(_ci(item, "createdate", "createDate", "created_at")),
            ))
        unknown = [r.profile_type for r in rows if r.agent is None]
        if unknown:
            # Not fatal: a new profile type is additive, and the three we map
            # still produce complete artifacts.
            logger.warning("smx_unknown_profile_types",
                           extra={"corp_no": corp_no, "profile_types": unknown})
        return rows

    def generate(self, corp_nos: list[int]) -> dict[str, Any]:
        """Trigger generation. Write operation — callers gate this on user intent."""
        if not corp_nos:
            raise SmxClientError("generate() needs at least one corp_no")
        return self._request("POST", "/AIAccountProfile/Generate",
                             json={"corpNos": [int(c) for c in corp_nos]}) or {}

    def package_types(self) -> list[str]:
        data = self._request("GET", "/AIAccountProfile/PackageTypes") or []
        return [str(x) for x in data] if isinstance(data, list) else []
