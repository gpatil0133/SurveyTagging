"""Tenant-level taggers.

Each module here defines one TenantTagger that reads TenantProfile and emits a
single high-level semantic tag. Runs once per tenant (not per survey) — the
orchestrator writes the combined output to
output/{tenant_id}/tenant_tags.json.

Keep tags semantic. No backend codes, no IDs, no widget payloads.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taggers.tenant.base import TenantTagger


def discover_tenant_taggers() -> list["TenantTagger"]:
    """Import every module in this package and collect their tagger instances.

    Each tagger module must expose a module-level `create_tagger()` factory.
    """
    instances: list[TenantTagger] = []
    package = importlib.import_module(__name__)
    for mod_info in pkgutil.iter_modules(package.__path__):
        if mod_info.name in ("base", "__init__"):
            continue
        module = importlib.import_module(f"{__name__}.{mod_info.name}")
        factory = getattr(module, "create_tagger", None)
        if factory is None:
            continue
        instances.append(factory())
    return instances
