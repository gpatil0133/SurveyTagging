"""Signal models extracted from response data, directories, and invitations."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field


class ResponseStats(BaseModel):
    """Aggregated statistics from survey response batch files."""

    total_responses: int = 0
    complete_count: int = 0
    partial_count: int = 0
    completion_rate: float = 0.0
    date_range: tuple[datetime, datetime] | None = None
    span_days: int = 0
    response_timestamps: list[datetime] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    median_assessment_time_seconds: float | None = None


class DirectorySignals(BaseModel):
    """Domain signals extracted from directory parquet schemas (column names only)."""

    directory_ids: list[int] = Field(default_factory=list)
    column_names: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)
    inferred_domains: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.column_names) == 0


class InvitationSignals(BaseModel):
    """Signals extracted from invitation/distribution data."""

    total_invitations: int = 0
    channel_distribution: dict[str, int] = Field(default_factory=dict)
    date_range: tuple[datetime, datetime] | None = None
    response_rate: float = 0.0

    @property
    def primary_channel(self) -> str | None:
        if not self.channel_distribution:
            return None
        return max(self.channel_distribution, key=self.channel_distribution.get)
