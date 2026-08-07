"""Load invitation/distribution signals from invitation_data.parquet."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

import sharefs

from models.signals import InvitationSignals

logger = logging.getLogger(__name__)


def load_invitation_signals(survey_dir: Path) -> InvitationSignals | None:
    """Load invitation data and extract distribution signals.

    Args:
        survey_dir: Path to the survey folder.

    Returns:
        InvitationSignals or None if no invitation data exists.
    """
    inv_file = survey_dir / "invitation_data.parquet"
    if not sharefs.exists(inv_file):
        return None

    try:
        with sharefs.open_file(inv_file, "rb") as fh:
            table = pq.read_table(fh)
    except Exception as e:
        logger.warning("invitation_read_error", extra={"file": str(inv_file), "error": str(e)})
        return None

    total = table.num_rows
    if total == 0:
        return None

    # Channel distribution
    channel_dist: dict[str, int] = {}
    if "Channel" in table.column_names:
        for val in table.column("Channel").to_pylist():
            channel = str(val or "Unknown").strip()
            channel_dist[channel] = channel_dist.get(channel, 0) + 1

    # Date range
    dates: list[datetime] = []
    for col_name in ["DeliveryDate", "ResponseDate"]:
        if col_name in table.column_names:
            for val in table.column(col_name).to_pylist():
                dt = _parse_inv_datetime(val)
                if dt:
                    dates.append(dt)

    date_range = None
    if dates:
        dates.sort()
        date_range = (dates[0], dates[-1])

    # Response rate
    responded = 0
    if "Response" in table.column_names:
        for val in table.column("Response").to_pylist():
            if str(val).lower() == "true":
                responded += 1

    return InvitationSignals(
        total_invitations=total,
        channel_distribution=channel_dist,
        date_range=date_range,
        response_rate=responded / total if total > 0 else 0.0,
    )


def _parse_inv_datetime(val: str | None) -> datetime | None:
    """Parse datetime from invitation data formats."""
    if not val or not isinstance(val, str):
        return None
    formats = [
        "%m/%d/%Y %I:%M:%S %p",   # 3/25/2026 3:57:00 AM
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val.strip(), fmt)
        except ValueError:
            continue
    return None
