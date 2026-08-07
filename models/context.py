"""Unified context model — the single object passed to all taggers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from models.overrides import ManualOverrides
from models.survey import QuestionContext, SurveyMeta
from models.signals import DirectorySignals, InvitationSignals, ResponseStats
from models.tenant_profile import TenantProfile
from models.tenant_canon import TenantCanon

if TYPE_CHECKING:
    from llm.embeddings import CanonEmbeddingIndex


class MatrixGroup(BaseModel):
    """A group of questions sharing the same questionNo and matrixgrouptitle."""

    question_no: int
    group_title: str
    question_ids: list[int] = Field(default_factory=list)
    size: int = 0


class ExistingTags(BaseModel):
    """Pre-populated sentiment/themes/emotions from the survey structure."""

    sentiment: str | None = None
    themes: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.sentiment or self.themes or self.emotions)


class UnifiedContext(BaseModel):
    """Complete context for tagging a single survey.

    Assembled by the context_assembler from all data sources.
    Passed to every tagger as read-only input.
    """

    # `canon_embeddings` carries a numpy array; allow arbitrary types so
    # Pydantic doesn't try to validate the dataclass.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tenant_id: int
    # Caller-supplied tenant hints. Only the ad-hoc `POST /api/tag` path fills
    # these; the disk pipeline leaves them empty and uses `tenant_profile`.
    overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    tenant_profile: TenantProfile | None = None
    # V5: tenant canon + embeddings, loaded once per tenant in the
    # orchestrator and attached here for every survey of that tenant.
    # Both are optional: tenants without a canon fall back to the
    # legacy industry-template path inside the prompt builder.
    # `tenant_canon` / `canon_embeddings` hold the CX canon; the `_ex` pair
    # holds the EX (employee-lifecycle) canon. Use `canon_for(project_type)`
    # to pick the right one for a survey.
    tenant_canon: TenantCanon | None = None
    canon_embeddings: Any | None = None  # llm.embeddings.CanonEmbeddingIndex
    tenant_canon_ex: TenantCanon | None = None
    canon_embeddings_ex: Any | None = None  # llm.embeddings.CanonEmbeddingIndex
    survey_meta: SurveyMeta
    questions: list[QuestionContext] = Field(default_factory=list)
    response_stats: ResponseStats | None = None
    directory_signals: DirectorySignals = Field(default_factory=DirectorySignals)
    invitation_signals: InvitationSignals | None = None
    has_linking: bool = False
    has_prepop: bool = False

    # Computed during context assembly
    question_groups: list[MatrixGroup] = Field(default_factory=list)
    piping_map: dict[int, list[str]] = Field(default_factory=dict)
    existing_tags: ExistingTags | None = None
    # V5: precomputed question_id -> nearest preceding section header title.
    # Populated by context_assembler in a single forward walk; used by the
    # per-question signature builder to avoid O(N²) lookups.
    section_for_qid: dict[int, str] = Field(default_factory=dict)

    @property
    def non_cm_questions(self) -> list[QuestionContext]:
        """Questions that are actual questions (not content messages)."""
        return [q for q in self.questions if not q.is_content_message]

    @property
    def has_responses(self) -> bool:
        return self.response_stats is not None and self.response_stats.total_responses > 0

    @property
    def has_nps(self) -> bool:
        return any(q.is_nps for q in self.questions)

    @property
    def has_csat(self) -> bool:
        return any(q.is_csat for q in self.questions)

    @property
    def cm_section_headers(self) -> list[str]:
        """Titles of CM questions that serve as section headers."""
        return [q.title for q in self.questions if q.is_content_message and q.title]

    def section_header_for(self, q: QuestionContext) -> str:
        """Nearest section header (CM question title) preceding `q`. "" if none."""
        return self.section_for_qid.get(q.question_id, "")

    def canon_for(self, project_type: str | None) -> tuple[TenantCanon | None, Any | None]:
        """Return (canon, embeddings) for a survey given its project_type.

        EX surveys ground journey stages against the employee-lifecycle canon;
        every other project type uses the CX canon. Falls back to the CX canon
        when the EX canon is missing so EX surveys stay runnable.
        """
        if (project_type or "").strip().upper() == "EX" and self.tenant_canon_ex is not None:
            return self.tenant_canon_ex, self.canon_embeddings_ex
        return self.tenant_canon, self.canon_embeddings
