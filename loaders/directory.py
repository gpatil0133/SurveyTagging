"""Load directory signals from parquet schemas (column names only — no data loaded)."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import sharefs

from models.signals import DirectorySignals

logger = logging.getLogger(__name__)

# Default domain keywords (loaded from config if available)
_DOMAIN_KEYWORDS: dict[str, list[str]] | None = None


def _get_domain_keywords(config_dir: Path | None = None) -> dict[str, list[str]]:
    """Load domain keyword mapping from scale_patterns.yaml or use defaults."""
    global _DOMAIN_KEYWORDS
    if _DOMAIN_KEYWORDS is not None:
        return _DOMAIN_KEYWORDS

    if config_dir:
        config_file = config_dir / "scale_patterns.yaml"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                _DOMAIN_KEYWORDS = data.get("domain_keywords", {})
                return _DOMAIN_KEYWORDS

    # Fallback defaults
    _DOMAIN_KEYWORDS = {
        "Healthcare": ["Patient", "Hospital", "Visit_Type", "Diagnosis", "OPD", "Doctor"],
        "Financial Services": ["Account_Number", "Branch_Code", "Transaction", "Loan", "Balance"],
        "Higher Education": ["Student_ID", "Grade", "Course", "Faculty", "Semester"],
        "K-12 Education": ["Student_ID", "Grade", "Parent_Name", "Teacher", "School"],
        "Hospitality / Travel": ["Hotel", "Room_Type", "Loyalty_Points", "Guest", "Reservation"],
        "Retail / E-commerce": ["Product", "Purchase", "Store", "Order", "Bike_Model"],
        "SaaS / Technology": ["License", "Subscription", "API", "Feature", "Ticket"],
    }
    return _DOMAIN_KEYWORDS


# ---------------------------------------------------------------------------
# Segment candidates
# ---------------------------------------------------------------------------
#
# A directory column can break survey results out by respondent ("CSAT by
# department", "eNPS by location") when it has few enough distinct values to make
# readable groups and is not an identifier. Two filters, both necessary:
#
#   cardinality  2..15 distinct values. One value groups nothing; past 15 the
#                cells get too thin to read, the same threshold is_segmentable
#                already uses for answer options.
#   not personal a name, an email, a date of birth or a record id is not a
#                segment, and its values must never be copied into
#                tagged_output.json — which is written to the share and read
#                widely. Cardinality alone would exclude most of them by
#                accident; this list excludes them on purpose.
#
# Values ARE read (that is the point — the consumer needs the vocabulary), so the
# deny-list is a privacy boundary, not a tidiness preference.

_SEGMENT_MAX_CARDINALITY = 15
_SEGMENT_MIN_CARDINALITY = 2

# Substring match, case-insensitive, on the column name.
_NEVER_A_SEGMENT = (
    "name", "email", "phone", "mobile", "address", "birth", "dob",
    "_id", "id_", "recno", "uid", "uuid", "ssn", "passport", "salary",
)

# Date-ish columns: the raw values are near-unique so cardinality already drops
# them, but a tenure/recency band is a genuinely useful segment. Naming them here
# lets the evidence say "there is a date here that could be banded" instead of
# silently dropping it. Banding itself needs a product decision on bucket edges.
_DATE_HINTS = ("date", "hire", "start", "joined", "tenure")


def _is_identifying(column: str) -> bool:
    low = column.lower().replace(" ", "_")
    return any(token in low for token in _NEVER_A_SEGMENT)


def load_segment_candidates(
    tenant_dir: Path,
    max_cardinality: int = _SEGMENT_MAX_CARDINALITY,
) -> dict[str, dict[str, list[str]]]:
    """`{directory_id: {attribute: [distinct values]}}` for the whole tenant.

    Reads the candidate columns' VALUES, unlike `load_directory_signals` which
    reads only names — so it is deliberately a separate call, made once per tenant
    run and cached on `DirectorySignals`. Cost is one parquet read per directory,
    projected to the surviving columns.

    Any failure yields `{}` for that directory: a tenant whose directory cannot be
    read must still get every other dimension.
    """
    dir_path = tenant_dir / "Directory"
    if not sharefs.exists(dir_path):
        return {}

    out: dict[str, dict[str, list[str]]] = {}
    for directory_id, files in _directory_files(dir_path).items():
        attributes: dict[str, list[str]] = {}
        for parquet_file in files:
            try:
                with sharefs.open_file(parquet_file, "rb") as fh:
                    schema = pq.read_schema(fh)
                wanted = [c for c in schema.names if not _is_identifying(c)]
                if not wanted:
                    continue
                with sharefs.open_file(parquet_file, "rb") as fh:
                    table = pq.read_table(fh, columns=wanted)
            except Exception as e:  # noqa: BLE001 — pyarrow raises its own types
                logger.warning("segment_candidates_read_failed",
                               extra={"file": str(parquet_file),
                                      "error": f"{type(e).__name__}: {e}"})
                continue

            near_misses: dict[str, int] = {}
            for column in table.column_names:
                values = {
                    str(v).strip() for v in table.column(column).to_pylist()
                    if v is not None and str(v).strip()
                }
                if not (_SEGMENT_MIN_CARDINALITY <= len(values) <= max_cardinality):
                    # The cap is a judgement, not a fact — 15 groups is where cells
                    # start getting too thin to read, but a tenant with 18 cities has
                    # a segment we are declining. Log what just missed so the number
                    # can be tuned from evidence instead of taste; `City` on the QA
                    # tenant is exactly this case.
                    if max_cardinality < len(values) <= max_cardinality * 4:
                        near_misses[column] = len(values)
                    continue
                merged = set(attributes.get(column, [])) | values
                if len(merged) <= max_cardinality:
                    attributes[column] = sorted(merged)

            if near_misses:
                logger.debug("segment_candidates_over_cap",
                             extra={"file": str(parquet_file),
                                    "cap": max_cardinality,
                                    "columns": near_misses})

        if attributes:
            out[str(directory_id)] = attributes

    logger.debug("segment_candidates_loaded",
                 extra={"tenant_dir": str(tenant_dir),
                        "directories": len(out),
                        "attributes": sum(len(a) for a in out.values())})
    return out


def _directory_files(dir_path: Path) -> dict[str, list[Path]]:
    """`{directory_id: [parquet files]}` across both layouts on the share.

    Flat `directory_{ID}.parquet` and subfolder `{ID}/batch_*.parquet` both occur,
    which is why `load_directory_signals` walks them twice; this is the same walk
    expressed once, keyed by the id the linking file uses.
    """
    found: dict[str, list[Path]] = {}
    for parquet_file in sharefs.glob(dir_path, "directory_*.parquet"):
        directory_id = parquet_file.stem.replace("directory_", "")
        found.setdefault(directory_id, []).append(parquet_file)
    for subdir in sharefs.iterdir(dir_path):
        if sharefs.is_dir(subdir):
            batches = list(sharefs.glob(subdir, "*.parquet"))
            if batches:
                found.setdefault(subdir.name, []).extend(batches)
    return found


def load_linked_directory_ids(survey_dir: Path) -> list[str]:
    """Directory ids this survey's responses are joined to, from
    `directory_linking.parquet`.

    That file is the reason segments are possible at all: it maps a ResponseUID to
    a DirectoryRecordMasterId, so a respondent's attributes can be attached to
    their answers. Without it the tenant's directory data describes people who
    cannot be tied to any response, and no segment is buildable.
    """
    path = survey_dir / "directory_linking.parquet"
    if not sharefs.exists(path):
        return []
    try:
        with sharefs.open_file(path, "rb") as fh:
            table = pq.read_table(fh, columns=["DirectoryId"])
    except Exception as e:  # noqa: BLE001
        logger.warning("directory_linking_read_failed",
                       extra={"path": str(path), "error": f"{type(e).__name__}: {e}"})
        return []
    ids = {str(v).strip() for v in table.column("DirectoryId").to_pylist()
           if v is not None and str(v).strip()}
    return sorted(ids)


def load_directory_signals(
    tenant_dir: Path,
    config_dir: Path | None = None,
) -> DirectorySignals:
    """Extract domain signals from directory parquet file schemas.

    Reads ONLY column names via pq.read_schema() — never loads data.
    Handles both directory patterns:
      - Flat: Directory/directory_{ID}.parquet
      - Subfolder: Directory/{ID}/batch_*.parquet

    Args:
        tenant_dir: Path to the tenant folder.
        config_dir: Path to config/ for domain keyword definitions.

    Returns:
        DirectorySignals with column names and inferred domains.
    """
    dir_path = tenant_dir / "Directory"
    if not sharefs.exists(dir_path):
        return DirectorySignals()

    all_columns: set[str] = set()
    directory_ids: list[int] = []

    # Pattern 1: Flat files — directory_{ID}.parquet
    for parquet_file in sharefs.glob(dir_path, "directory_*.parquet"):
        try:
            dir_id_str = parquet_file.stem.replace("directory_", "")
            directory_ids.append(int(dir_id_str))
            with sharefs.open_file(parquet_file, "rb") as fh:
                schema = pq.read_schema(fh)
            all_columns.update(schema.names)
        except Exception as e:
            logger.warning("directory_schema_error", extra={"file": str(parquet_file), "error": str(e)})

    # Pattern 2: Subfolders — {ID}/batch_*.parquet
    for subdir in sharefs.iterdir(dir_path):
        if sharefs.is_dir(subdir):
            try:
                dir_id = int(subdir.name)
                if dir_id not in directory_ids:
                    directory_ids.append(dir_id)
            except ValueError:
                continue

            for batch_file in sharefs.glob(subdir, "batch_*.parquet"):
                try:
                    with sharefs.open_file(batch_file, "rb") as fh:
                        schema = pq.read_schema(fh)
                    all_columns.update(schema.names)
                except Exception as e:
                    logger.warning("directory_batch_schema_error",
                                   extra={"file": str(batch_file), "error": str(e)})

    if not all_columns:
        return DirectorySignals(directory_ids=sorted(directory_ids))

    # Extract domain keywords and infer industry domains
    domain_keywords_map = _get_domain_keywords(config_dir)
    matched_keywords: list[str] = []
    inferred_domains: dict[str, int] = {}  # domain → match count

    column_names_lower = [c.lower().replace(" ", "_") for c in all_columns]

    for domain, keywords in domain_keywords_map.items():
        matches = 0
        for keyword in keywords:
            keyword_lower = keyword.lower().replace(" ", "_")
            # Count how many columns contain this keyword (not just presence)
            col_matches = sum(1 for col in column_names_lower if keyword_lower in col)
            if col_matches > 0:
                matched_keywords.append(keyword)
                matches += col_matches  # Weight by actual column count
        if matches >= 2:  # Need at least 2 matching columns to infer a domain
            inferred_domains[domain] = matches

    # Sort domains by match count (strongest signal first)
    sorted_domains = sorted(inferred_domains, key=inferred_domains.get, reverse=True)

    return DirectorySignals(
        directory_ids=sorted(directory_ids),
        column_names=sorted(all_columns),
        domain_keywords=sorted(set(matched_keywords)),
        inferred_domains=sorted_domains,
        segment_candidates=load_segment_candidates(tenant_dir),
    )
