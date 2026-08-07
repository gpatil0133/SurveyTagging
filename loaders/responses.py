"""Load response statistics from batch_*.parquet files."""

from __future__ import annotations

import logging
import statistics
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

import sharefs

from models.signals import ResponseStats

logger = logging.getLogger(__name__)

# Metadata columns to read (avoid loading actual response data)
_META_COLUMNS = [
    "StartTime", "EndTime", "PaticipationStatus",
    "Language", "AssessmentTime", "Complete",
]


def load_response_stats(survey_dir: Path) -> ResponseStats | None:
    """Load and aggregate response metadata from batch parquet files.

    Only reads metadata columns — never loads actual answer data.

    Args:
        survey_dir: Path to the survey folder containing batch_*.parquet files.

    Returns:
        ResponseStats or None if no batch files exist.
    """
    batch_files = sorted(sharefs.glob(survey_dir, "batch_*.parquet"))
    if not batch_files:
        return None

    all_timestamps: list[datetime] = []
    all_languages: set[str] = set()
    total = 0
    complete = 0
    partial = 0
    assessment_times: list[float] = []

    for batch_file in batch_files:
        try:
            # One handle per file, rewound between reads: passing a path to
            # each pq.* call would be a fresh open + round trip over SMB.
            with sharefs.open_file(batch_file, "rb") as fh:
                # Read only the columns we need
                schema = pq.read_schema(fh)
                available_cols = [c for c in _META_COLUMNS if c in schema.names]

                if not available_cols:
                    # Fallback: just count rows
                    fh.seek(0)
                    meta = pq.read_metadata(fh)
                    total += meta.num_rows
                    continue

                fh.seek(0)
                table = pq.read_table(fh, columns=available_cols)
            total += table.num_rows

            # Parse timestamps
            if "StartTime" in table.column_names:
                for val in table.column("StartTime").to_pylist():
                    ts = _parse_datetime(val)
                    if ts:
                        all_timestamps.append(ts)

            # Count completion status
            if "PaticipationStatus" in table.column_names:
                for val in table.column("PaticipationStatus").to_pylist():
                    status = str(val or "").strip().lower()
                    if status == "complete":
                        complete += 1
                    elif status in ("partial", "incomplete"):
                        partial += 1
            elif "Complete" in table.column_names:
                for val in table.column("Complete").to_pylist():
                    if str(val).lower() == "true":
                        complete += 1
                    else:
                        partial += 1

            # Languages
            if "Language" in table.column_names:
                for val in table.column("Language").to_pylist():
                    if val:
                        all_languages.add(str(val).strip())

            # Assessment time
            if "AssessmentTime" in table.column_names:
                for val in table.column("AssessmentTime").to_pylist():
                    secs = _parse_assessment_time(val)
                    if secs is not None:
                        assessment_times.append(secs)

        except Exception as e:
            logger.warning("batch_read_error", extra={"file": str(batch_file), "error": str(e)})
            continue

    if total == 0:
        return None

    # Sort timestamps for temporal analysis
    all_timestamps.sort()

    date_range = None
    span_days = 0
    if len(all_timestamps) >= 2:
        date_range = (all_timestamps[0], all_timestamps[-1])
        span_days = (all_timestamps[-1] - all_timestamps[0]).days

    return ResponseStats(
        total_responses=total,
        complete_count=complete,
        partial_count=partial,
        completion_rate=complete / total if total > 0 else 0.0,
        date_range=date_range,
        span_days=span_days,
        response_timestamps=all_timestamps,
        languages=sorted(all_languages),
        median_assessment_time_seconds=(
            statistics.median(assessment_times) if assessment_times else None
        ),
    )


def _parse_datetime(val: str | None) -> datetime | None:
    """Parse datetime from various formats found in response data."""
    if not val or not isinstance(val, str):
        return None

    # Common formats in the data
    formats = [
        "%d-%b-%Y %I:%M %p",       # 30-Sep-2024 03:45 PM
        "%Y-%m-%d %H:%M:%S.%f",    # 2024-09-30 15:49:00.000
        "%Y-%m-%dT%H:%M:%S.%f",    # 2024-09-30T15:49:00.000
        "%Y-%m-%dT%H:%M:%S",       # 2024-09-30T15:49:00
        "%m/%d/%Y %I:%M:%S %p",    # 3/25/2026 3:57:00 AM
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val.strip(), fmt)
        except ValueError:
            continue

    return None


def _parse_assessment_time(val: str | None) -> float | None:
    """Parse assessment time (HH:MM:SS or MM:SS format) to seconds."""
    if not val or not isinstance(val, str):
        return None
    try:
        parts = val.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return None
