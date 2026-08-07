"""Base class for tenant-level taggers.

Unlike ProjectTagger / QuestionTagger, TenantTagger runs once per tenant and
does not take a per-survey UnifiedContext. It reads TenantProfile and emits one
TagResult representing a tenant-shape signal the backend dashboard writer can
act on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from models.tags import TagResult
from models.tenant_profile import TenantProfile


class TenantTagger(ABC):
    """Abstract base for tenant-shape taggers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tagger ID, e.g., 'tenant.compliance_posture'."""
        ...

    @property
    @abstractmethod
    def tag_dimension(self) -> str:
        """Taxonomy dimension key, e.g., 'compliance_posture'."""
        ...

    @property
    def source_type(self) -> Literal["deterministic", "hybrid"]:
        return "deterministic"

    @abstractmethod
    def tag(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> TagResult:
        """Assign a tenant-level tag.

        Returns a TagResult. When the tenant_profile is missing or the
        relevant section is absent, return a TagResult with a sensible
        fallback value (typically "N/A" or []) at low confidence rather
        than raising.
        """
        ...
