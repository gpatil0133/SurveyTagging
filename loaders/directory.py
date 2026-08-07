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
    )
