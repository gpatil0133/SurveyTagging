"""Tag result models and the tag accumulator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TagResult(BaseModel):
    """Result of a single tag assignment."""

    value: Any = None  # str, list[str], or None
    source: Literal["deterministic", "statistical", "hybrid", "llm", "heuristic"] = "deterministic"
    confidence: float = 1.0
    # Why this value was assigned, for non-LLM sources. Non-LLM taggers build
    # this with `models.evidence` helpers → a typed dict
    # {type, rule_id, stage, detail, inputs?, quote?, measure?, components?}.
    # `str` stays accepted so pre-existing artifacts still validate on read;
    # use `models.evidence.detail_of()` rather than assuming either shape.
    evidence: str | dict[str, Any] | None = None
    # Free-text model rationale, for `source="llm"` tags only. The two fields
    # are complementary, never redundant: `source` tells a consumer which to read.
    # V7.1: this is the rationale for THIS dimension, not a per-question blob —
    # `llm_enhance` stamps the model's per-dimension `why` line here and falls
    # back to the question/survey summary only when the model gave no line.
    reasoning: str | None = None
    # V7.1: the assignment this one displaced, when an LLM answer overrode an
    # earlier deterministic/statistical tag: {value, source, confidence,
    # evidence?}. Without it the rule's evidence is simply overwritten and the
    # output cannot answer "a rule looked at this — what did it say?".
    superseded: dict[str, Any] | None = None
    # V5: low_confidence_assigned is for journey_stage / sub_stage_name when
    # the LLM was uncertain or punted to a top-K candidate. The value is set
    # (NEVER null) so downstream coverage can count it; consumers may flag
    # for review based on the status.
    status: Literal["assigned", "skipped", "failed", "pending_llm", "low_confidence_assigned"] = "assigned"
    failure_reason: str | None = None
    apply_method: str = "System-applied"
    # V5: structured metadata about why the LLM made this decision —
    # candidates ranked by embedding similarity, the LLM's stated confidence,
    # and the natural-language evidence sentence. Surfaces in tagged_output
    # for downstream coverage instrumentation; None for non-eligible questions.
    coverage_metadata: dict[str, Any] | None = None

    @property
    def is_assigned(self) -> bool:
        # V5: low_confidence_assigned counts as assigned for downstream consumers.
        # The journey-assembly partition treats them as kept (flagged for review),
        # not as ambiguous.
        return self.status in ("assigned", "low_confidence_assigned") and self.value is not None


class TagAccumulator:
    """Mutable container for accumulating tags during pipeline execution.

    Taggers write to this; downstream taggers read from it.
    Thread-safe for single-survey sequential processing.
    """

    def __init__(self) -> None:
        self._project_tags: dict[str, TagResult] = {}
        self._question_tags: dict[int, dict[str, TagResult]] = {}
        self._failures: dict[str, str] = {}

    # --- Project-level ---

    def get_project_tag(self, dimension: str) -> TagResult | None:
        return self._project_tags.get(dimension)

    def set_project_tag(self, dimension: str, result: TagResult) -> None:
        self._project_tags[dimension] = result

    def get_project_tag_value(self, dimension: str) -> Any:
        tag = self._project_tags.get(dimension)
        return tag.value if tag and tag.is_assigned else None

    # --- Question-level ---

    def get_question_tag(self, question_id: int, dimension: str) -> TagResult | None:
        return self._question_tags.get(question_id, {}).get(dimension)

    def set_question_tag(self, question_id: int, dimension: str, result: TagResult) -> None:
        if question_id not in self._question_tags:
            self._question_tags[question_id] = {}
        self._question_tags[question_id][dimension] = result

    def get_question_tag_value(self, question_id: int, dimension: str) -> Any:
        tag = self.get_question_tag(question_id, dimension)
        return tag.value if tag and tag.is_assigned else None

    # --- Failure tracking ---

    def mark_failed(self, dimension: str, reason: str) -> None:
        self._failures[dimension] = reason

    # --- Export ---

    @property
    def project_tags(self) -> dict[str, TagResult]:
        return dict(self._project_tags)

    @property
    def question_tags(self) -> dict[int, dict[str, TagResult]]:
        return dict(self._question_tags)

    @property
    def failures(self) -> dict[str, str]:
        return dict(self._failures)


class TaggedQuestion(BaseModel):
    """Output model for a tagged question."""

    question_id: int
    question_no: int
    question_title_preview: str = ""
    question_text: str = ""
    is_content_message: bool = False
    # Raw platform signals needed downstream for journey-eligibility checks
    # without reloading survey_structure.json.
    rs_type: int = 0
    is_custom_metric: bool = False
    tags: dict[str, Any] = Field(default_factory=dict)


class TaggedSurvey(BaseModel):
    """Complete tagged output for a single survey."""

    # The taxonomy generation that produced the artifact. It tracks the SHAPE of
    # this file, not every taxonomy release: V6 and V7 added and removed
    # dimensions without changing how one is represented, so no 6.0 or 7.0 was
    # ever issued and a reader correlating an artifact to a taxonomy should not
    # go looking for them.
    #
    # 8.0 — the widget-API alignment. Three shape changes, in the order a
    # consumer will hit them:
    #   * `segment_dimensions` entries are `{label, question_id}` objects, not
    #     bare strings. The one genuinely BREAKING change: a reader that joins
    #     or renders these gets objects where it expected text.
    #   * `favorable_options` is the first tag whose value is an OBJECT
    #     (`{positive, negative, neutral}` id lists) rather than a string or a
    #     list of strings. Value handling that assumed those two shapes needs a
    #     third branch — see static/render.js::fmtValue.
    #   * Six new question dimensions appear (platform_metric,
    #     favorable_options, trend_granularity, widget_footprint,
    #     preferred_segments, display_label), and the chart vocabulary in
    #     widget_compatibility / visualization_type was rewritten onto the
    #     platform's own names. Additive, but no V7 value survives unchanged in
    #     those two.
    #
    # 5.0 — atomic journey assignment (stage + sub_stage emitted together by the
    # LLM), canon-namespace journey_stage values, low_confidence_assigned
    # status, coverage_metadata on per-question tags.
    schema_version: str = "8.0"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tenant_id: int
    survey_no: int
    zarca_id: int
    survey_name: str = ""
    project_tags: dict[str, Any] = Field(default_factory=dict)
    question_tags: list[TaggedQuestion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TenantTags(BaseModel):
    """Tenant-level shape tags persisted once per tenant.

    Produced by `taggers/tenant/*` runners using TenantProfile.
    Lives at output/{tenant_id}/tenant_tags.json so the backend dashboard writer
    can read tenant-shape direction (compliance posture, workforce signature,
    key touchpoints, etc.) without re-deriving from Parallel.ai artifacts per
    survey.
    """

    schema_version: str = "1.0"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tenant_id: int
    # Each entry: {value, source, confidence, evidence?} — same shape used by
    # TaggedSurvey.project_tags so consumers can treat the formats uniformly.
    tags: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
