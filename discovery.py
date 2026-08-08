"""Single source of tenant / survey discovery + path building over the data root.

Layout (one root holds inputs and generated artifacts side by side):
    {root}/{tenant_id}/SurveyData/{survey_no}/survey_structure.json   (in)
    {root}/{tenant_id}/SurveyData/{survey_no}/tagged_output.json      (out)
    {root}/{tenant_id}/tenant_tags.json                              (out)

Path *algebra* is pathlib; every path *query* goes through `sharefs`. That split
is not cosmetic: on Windows plain `Path.iterdir()/.exists()` on a UNC root is
served by the OS SMB redirector using the process's own logon, so it happens to
work on a developer workstation and returns empty for a service account that has
no share credentials — the same root that `sharefs` reads fine via
SURVEY_TAGGER_IMAGE_USER. That reads downstream as "this tenant has no surveys"
(HTTP 404) rather than as an auth failure. Off Windows it cannot open a UNC path
at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import sharefs


def _as_int(name: str) -> int | None:
    try:
        return int(name)
    except ValueError:
        return None


def tenant_dir(data_dir: Path, tenant_id: int) -> Path:
    return Path(data_dir) / str(tenant_id)


def survey_dir(data_dir: Path, tenant_id: int, survey_no: int) -> Path:
    return Path(data_dir) / str(tenant_id) / "SurveyData" / str(survey_no)


def list_tenant_ids(data_dir: Path) -> list[int]:
    """Numeric tenant dirs that contain a SurveyData/ subfolder (sorted)."""
    try:
        entries = sharefs.list_dirs(data_dir)
    except OSError:
        return []
    out: list[int] = []
    for item in entries:
        tid = _as_int(item.name)
        if tid is not None and sharefs.exists(item / "SurveyData"):
            out.append(tid)
    return sorted(out)


def list_survey_nos(data_dir: Path, tenant_id: int) -> list[int]:
    """Survey dirs with a survey_structure.json under a tenant (sorted)."""
    base = Path(data_dir) / str(tenant_id) / "SurveyData"
    try:
        entries = sharefs.list_dirs(base)
    except OSError:
        return []
    out: list[int] = []
    for item in entries:
        sno = _as_int(item.name)
        if sno is not None and sharefs.exists(item / "survey_structure.json"):
            out.append(sno)
    return sorted(out)


def survey_exists(data_dir: Path, tenant_id: int, survey_no: int) -> bool:
    return sharefs.exists(survey_dir(data_dir, tenant_id, survey_no) / "survey_structure.json")


# Generated artifacts that live inside the survey dir alongside its inputs.
# The change detector must exclude these or writing the output would perturb
# the survey's own input hash — see pipeline/change_detector.py.
TAGGED_OUTPUT_FILE = "tagged_output.json"


def tagged_output_path(root: Path, tenant_id: int, survey_no: int) -> Path:
    """Where a survey's tags are written: next to its own inputs on the share."""
    return survey_dir(root, tenant_id, survey_no) / TAGGED_OUTPUT_FILE


def _survey_title(structure_file: Path, fallback: str) -> str:
    try:
        raw = json.loads(sharefs.read_bytes(structure_file).decode("utf-8-sig"))
        sd = (raw.get("SurveyData", [raw])[0]
              if isinstance(raw.get("SurveyData"), list) else raw)
        return sd.get("surveyTitle") or fallback
    except Exception:
        return fallback


def probe_root(data_dir: Path) -> dict:
    """Is the data root reachable right now?

    Every other function here swallows OSError and returns an empty list, which
    makes "the share is down / credentials expired" indistinguishable from "this
    tenant has no surveys". The UI needs to tell those apart, so this reports the
    failure instead of hiding it.

    Goes through `sharefs` rather than `Path.is_dir` so that a UNC root is
    checked over the same authenticated SMB session the readers use — and so
    that this is the first thing to open that session, since startup no longer
    connects eagerly.
    """
    return sharefs.probe(data_dir)


def discover_catalog(data_dir: Path) -> list[dict]:
    """List every tenant + its surveys (with title + has_responses) on disk."""
    result: list[dict] = []
    for tid in list_tenant_ids(data_dir):
        surveys = []
        for sno in list_survey_nos(data_dir, tid):
            sdir = survey_dir(data_dir, tid, sno)
            title = _survey_title(sdir / "survey_structure.json", f"Survey #{sno}")
            surveys.append({
                "survey_no": sno,
                "title": title,
                "has_responses": bool(sharefs.glob(sdir, "batch_*.parquet")),
            })
        if surveys:
            result.append({"tenant_id": tid, "surveys": surveys})
    return result
