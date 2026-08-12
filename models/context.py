"""Unified context model — the single object passed to all taggers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from models.overrides import ManualOverrides
from models.survey import QuestionContext, SurveyMeta
from models.signals import DirectorySignals, InvitationSignals, ResponseStats
from models.tenant_profile import TenantProfile
from models.tenant_canon import TenantCanon
from models.journey import ProfileJourney

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
    # Journey source (v8): read straight off `tenant_profile/` once per tenant
    # in the orchestrator and attached here for every survey of that tenant.
    # `profile_journey` / `journey_index` hold CX; the `_ex` pair holds the
    # employee lifecycle. Use `journey_for(project_type)` to pick per survey.
    # Both are optional — a tenant with no profile gets no journey tags rather
    # than generic ones.
    profile_journey: ProfileJourney | None = None
    journey_index: Any | None = None  # llm.profile_journey.JourneyIndex
    profile_journey_ex: ProfileJourney | None = None
    journey_index_ex: Any | None = None  # llm.profile_journey.JourneyIndex
    # PARKED (v5-v7): the tenant-canon layer. Left on the model so previously
    # persisted artifacts and existing callers still construct, but nothing in
    # the pipeline populates or reads these now — the journey dimensions are
    # sourced from the profile above. See `llm/tenant_canon.py`.
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
        """Section-header titles, in survey order, without repeats. Sourced from
        the CM questions the loader folded into `section_header`."""
        seen: list[str] = []
        for q in self.questions:
            if q.section_header and q.section_header not in seen:
                seen.append(q.section_header)
        return seen

    def section_header_for(self, q: QuestionContext) -> str:
        """Nearest section header (CM question title) preceding `q`. "" if none."""
        return self.section_for_qid.get(q.question_id, "")

    def journey_for(self, project_type: str | None) -> tuple[ProfileJourney | None, Any | None]:
        """Return (journey, index) for a survey given its project_type.

        EX surveys ground against the employee lifecycle; every other project
        type uses the CX journey. Falls back to CX when the EX side of the
        profile is missing, so an EX survey under a CX-only profile still gets
        placed rather than dropped.
        """
        if (project_type or "").strip().upper() == "EX" and self.profile_journey_ex is not None:
            return self.profile_journey_ex, self.journey_index_ex
        return self.profile_journey, self.journey_index

    def canon_for(self, project_type: str | None) -> tuple[TenantCanon | None, Any | None]:
        """PARKED — the canon layer is no longer populated. Kept so existing
        callers and tests resolve; returns whatever was passed in, which in the
        live pipeline is always (None, None). Use `journey_for` instead.
        """
        if (project_type or "").strip().upper() == "EX" and self.tenant_canon_ex is not None:
            return self.tenant_canon_ex, self.canon_embeddings_ex
        return self.tenant_canon, self.canon_embeddings
