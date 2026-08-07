"""Key EX lifecycle stages tagger.

Extracts a flat list of semantic lifecycle-stage names from
tenant_profile.ex_lifecycle_stages. Backend aligns EX widgets with these
stages (Recruitment / Onboarding / Development / Retention / Exit, etc.).

Output: list[str] of Title-Cased stage names (no IDs, no codes).
"""

from __future__ import annotations

from typing import Any

from models import evidence as ev
from models.tags import TagResult
from models.tenant_profile import TenantProfile
from taggers.tenant.base import TenantTagger


def _clean(label: str) -> str:
    return " ".join(label.strip().split()).title()


def _collect_stages(stages: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name") or stage.get("stage_name") or stage.get("stage")
        if isinstance(name, str) and name.strip():
            seen[_clean(name)] = None
    return list(seen.keys())


class KeyExLifecycleStagesTagger(TenantTagger):
    name = "tenant.key_ex_lifecycle_stages"
    tag_dimension = "key_ex_lifecycle_stages"
    source_type = "deterministic"

    def tag(
        self,
        tenant_id: int,
        tenant_profile: TenantProfile | None,
    ) -> TagResult:
        if tenant_profile is None or not tenant_profile.has_ex:
            return TagResult(
                value=[],
                source="deterministic",
                confidence=0.40,
                evidence=ev.fallback(
                    "tenant.ex_stages.no_ex_profile",
                    "No EX intelligence artifact on disk, so there were no lifecycle "
                    "stages to extract. Empty list is 'unknown', not 'no stages'.",
                    inputs={"has_ex": False},
                ),
            )

        raw = tenant_profile.ex_lifecycle_stages
        stages = _collect_stages(raw)
        if not stages:
            return TagResult(
                value=[],
                source="deterministic",
                confidence=0.50,
                evidence=ev.profile(
                    "tenant.ex_stages.none_extracted",
                    f"The lifecycle analysis holds {len(raw)} raw entr(ies) but none "
                    "carried a usable stage name under name / stage_name / stage.",
                    field="ex.lifecycle_analysis.stages",
                    inputs={"raw_entry_count": len(raw)},
                ),
            )

        conf_label = (
            tenant_profile.ex_lifecycle_confidence
            or tenant_profile.ex_confidence
        )
        conf_base = {"High": 0.90, "Medium": 0.80, "Low": 0.65}.get(conf_label, 0.70)

        return TagResult(
            value=stages,
            source="deterministic",
            confidence=conf_base,
            evidence=ev.profile(
                "tenant.ex_stages.extracted",
                f"Read {len(stages)} named lifecycle stage(s) straight off the EX "
                f"profile; confidence mirrors the agent's own rating of that section "
                f"({conf_label or 'unrated'}).",
                field="ex.lifecycle_analysis.stages",
                inputs={"stage_count": len(stages),
                        "raw_entry_count": len(raw),
                        "agent_confidence": conf_label or "unknown"},
                quote=", ".join(stages[:8]),
            ),
        )


def create_tagger() -> KeyExLifecycleStagesTagger:
    return KeyExLifecycleStagesTagger()
