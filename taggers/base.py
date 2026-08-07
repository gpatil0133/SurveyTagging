"""Base tagger protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

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
    """Base class for question-level taggers."""

    @property
    def level(self) -> Literal["project", "question"]:
        return "question"

    @abstractmethod
    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        """Assign a question-level tag for a single question."""
        ...
