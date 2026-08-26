"""Base tagger protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult


class BaseTagger(ABC):
    """Abstract base for all taggers (project-level and question-level).

    Each tagger is a self-contained plugin that:
    - Declares its metadata (name, dimension, level, stage, dependencies)
    - Implements a tag() method for project-level or tag_question() for question-level
    - Reads from UnifiedContext (immutable) and TagAccumulator (read prior tags)
    - Returns TagResult(s)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tagger ID, e.g., 'project.project_type'."""
        ...

    @property
    @abstractmethod
    def tag_dimension(self) -> str:
        """Taxonomy dimension key, e.g., 'project_type'."""
        ...

    @property
    @abstractmethod
    def level(self) -> Literal["project", "question"]:
        """Whether this tagger produces project-level or question-level tags."""
        ...

    @property
    @abstractmethod
    def stage(self) -> int:
        """Execution stage (1-5). Lower stages run first."""
        ...

    @property
    def depends_on(self) -> list[str]:
        """Other tagger names this needs to have run first."""
        return []

    @property
    def source_type(self) -> Literal["deterministic", "statistical", "hybrid", "llm", "heuristic"]:
        """Classification method used by this tagger."""
        return "deterministic"


# NOTE: tenant-level taggers do NOT derive from these. `taggers/tenant/base.py`
# is a deliberately separate protocol — `tag(tenant_id, tenant_profile)` with no
# survey, no question loop and no accumulator, run once per tenant by
# `PipelineOrchestrator._tag_tenant` rather than through the registry. Adding a
# tenant dimension means editing that file; adding a project or question
# dimension means subclassing one of the two below.


class ProjectTagger(BaseTagger):
    """Base class for project-level taggers."""

    @property
    def level(self) -> Literal["project", "question"]:
        return "project"

    @abstractmethod
    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        """Assign a project-level tag."""
        ...


class QuestionTagger(BaseTagger):
    """Base class for question-level taggers.

    Subclasses implement `_tag_question`; the pipeline calls `tag_question`, which
    handles the one case every dimension answers identically first — see below.
    """

    # Content messages are page text, not questions: 22 of the 24 dimensions have
    # nothing to say about them and returned the same three-line skip, restated in
    # every file. Restating it is how one tagger ends up disagreeing with the rest
    # about what a CM row looks like, so the skip lives here once.
    #
    # `skips_content_messages = False` opts out for the two dimensions that
    # genuinely tag them (data_sensitivity: a CM collects nothing, so it is
    # Anonymous-safe; flow_experience: a welcome or section header is a real part
    # of the respondent's experience). Those see CM rows in `_tag_question`.
    skips_content_messages: bool = True

    # Value carried by the skip. `None` for scalar dimensions, `[]` for
    # multi-label ones, so a consumer iterating a list-valued tag never has to
    # None-check it.
    skip_value: Any = None

    @property
    def level(self) -> Literal["project", "question"]:
        return "question"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        """Tag one question. Do not override — implement `_tag_question`."""
        if question.is_content_message and self.skips_content_messages:
            return TagResult(
                value=self.skip_value, source="deterministic", status="skipped",
                evidence=ev.content_message(self.tag_dimension, stage=self.stage),
            )
        return self._tag_question(context, question, accumulator)

    @abstractmethod
    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        """Assign a question-level tag for a single, non-CM question."""
        ...
