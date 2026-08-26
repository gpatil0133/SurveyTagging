"""Unified context model — the single object passed to all taggers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.overrides import ManualOverrides
from models.survey import QuestionContext, SurveyMeta
from models.signals import (
    DirectorySignals,
    InvitationSignals,
    ResponseStats,
    VerbatimSignals,
)
from models.tenant_profile import TenantProfile
from models.journey import ProfileJourney


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tenant_id: int
    # Caller-supplied tenant hints. Only the ad-hoc `POST /api/tag` path fills
    # these; the disk pipeline leaves them empty and uses `tenant_profile`.
    overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    tenant_profile: TenantProfile | None = None
    # Journey source (v8): read straight off `tenant_profile/` once per tenant
    # in the orchestrator and attached here for every survey of that tenant.
    # `profile_journey` holds CX, `profile_journey_ex` the employee lifecycle.
    # Use `journey_for(project_type)` to pick per survey. Both are optional — a
    # tenant with no profile gets no journey tags rather than generic ones.
    #
    # V9 dropped the paired `journey_index` fields: the embedding index they
    # carried no longer exists, and the journey itself is what the prompt inlines.
    profile_journey: ProfileJourney | None = None
    profile_journey_ex: ProfileJourney | None = None
    survey_meta: SurveyMeta
    questions: list[QuestionContext] = Field(default_factory=list)
    response_stats: ResponseStats | None = None
    directory_signals: DirectorySignals = Field(default_factory=DirectorySignals)
    invitation_signals: InvitationSignals | None = None
    has_linking: bool = False
    has_prepop: bool = False
    # Directory ids this survey's responses are actually joined to
    # (`directory_linking.parquet`). A tenant may hold several directories — a
    # customer list and an employee list — and only the linked one describes THIS
    # survey's respondents.
    linked_directory_ids: list[str] = Field(default_factory=list)
    # `{question_id: VerbatimSignals}` from `survey_response_data.parquet`: what
    # the platform's text analytics already produced per open-text question.
    verbatim_signals: dict[int, VerbatimSignals] = Field(default_factory=dict)

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
    def respondent_segments(self) -> dict[str, list[str]]:
        """Segmentable respondent attributes for THIS survey.

        The tenant's candidates narrowed to the directories the survey is linked
        to, so an EX survey does not offer to break results out by a customer
        attribute. Empty when nothing links the responses to a directory record —
        which is the honest answer, not a shortage of data: without
        `directory_linking.parquet` no attribute can reach a response.
        """
        if not self.linked_directory_ids:
            return {}
        merged: dict[str, list[str]] = {}
        for directory_id in self.linked_directory_ids:
            for attribute, values in (
                self.directory_signals.segment_candidates.get(directory_id, {}).items()
            ):
                if attribute in merged:
                    merged[attribute] = sorted(set(merged[attribute]) | set(values))
                else:
                    merged[attribute] = list(values)
        return merged

    def verbatim_for(self, question_id: int) -> VerbatimSignals | None:
        """Text-analytics enrichment for one question, or None."""
        return self.verbatim_signals.get(question_id)

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

    def journey_for(self, project_type: str | None) -> ProfileJourney | None:
        """Return the journey a survey should be placed against.

        EX surveys ground against the employee lifecycle; every other project
        type uses the CX journey. Falls back to CX when the EX side of the
        profile is missing, so an EX survey under a CX-only profile still gets
        placed rather than dropped.
        """
        if (project_type or "").strip().upper() == "EX" and self.profile_journey_ex is not None:
            return self.profile_journey_ex
        return self.profile_journey
