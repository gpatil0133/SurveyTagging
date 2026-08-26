"""Signal models extracted from response data, directories, and invitations."""

from __future__ import annotations

from datetime import datetime

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
    """Domain signals extracted from the tenant's directory parquet files."""

    directory_ids: list[int] = Field(default_factory=list)
    column_names: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)
    inferred_domains: list[str] = Field(default_factory=list)
    # `{directory_id: {attribute: [distinct values]}}` for the low-cardinality,
    # non-identifying attributes only — the ones a report can break results out
    # by. Keyed by directory id as a string because that is how
    # `directory_linking.parquet` spells it, and matching there matters more than
    # the type being pretty. See `loaders.directory.load_segment_candidates`.
    segment_candidates: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.column_names) == 0


class VerbatimSignals(BaseModel):
    """What the platform's text analytics produced for ONE open-text question.

    Counts and label names only — never the answers themselves. `n_analyzed` is
    the number of verbatim rows for the question; the rest say how many of them
    carry each kind of enrichment, which is what decides whether a sentiment or
    theme widget has anything to render.
    """

    question_id: int
    n_analyzed: int = 0
    n_sentiment: int = 0
    n_sentiment_scored: int = 0
    n_themes: int = 0
    n_emotions: int = 0
    n_topics: int = 0
    n_action_plans: int = 0
    top_themes: list[str] = Field(default_factory=list)
    top_emotions: list[str] = Field(default_factory=list)
    top_topics: list[str] = Field(default_factory=list)

    @property
    def available(self) -> list[str]:
        """The enrichments actually present, as taxonomy values.

        Presence is "at least one row has it", not "all rows do": a partially
        scored question still supports the widget, and the counts are on the
        evidence for anyone who needs the coverage.
        """
        out = []
        if self.n_sentiment:
            out.append("Sentiment")
        if self.n_sentiment_scored:
            out.append("Sentiment Score")
        if self.n_themes:
            out.append("Themes")
        if self.n_emotions:
            out.append("Emotions")
        if self.n_topics:
            out.append("Topics")
        if self.n_action_plans:
            out.append("Action Plans")
        return out


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
