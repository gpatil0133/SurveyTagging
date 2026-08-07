"""Typed evidence builders — the "why" behind a non-LLM tag assignment.

Every deterministic / statistical / hybrid / heuristic tagger explains itself
through one of these builders instead of a bare sentence. The result is a plain
dict stored on `TagResult.evidence`, so it survives `model_dump()` into
`tagged_output.json` / `tenant_tags.json` untouched and is machine-readable
downstream (filter by `rule_id`, group by `type`, chart `inputs`).

LLM-sourced tags do NOT use these builders — they carry a free-text
`TagResult.reasoning` string produced by the model itself. `source` on the tag
already tells consumers which of the two to expect.

Shape (matches what `static/render.js::evidenceParts` renders):

    {
      "type":      "rule" | "statistic" | "hybrid" | "profile" | "fallback",
      "rule_id":   "metric.rs_type_nps",     # stable, greppable, dot-namespaced
      "stage":     3,                        # pipeline stage that produced it
      "detail":    "one plain sentence a human reads first",
      "inputs":    {"rs_type": 2},           # the signals the rule actually read
      "quote":     "...",                    # verbatim source text, when any
      "measure":   "response_gap_days",      # statistic only
      "observed":  31.4,                     # statistic only
      "threshold": 25,                       # statistic only
      "components": [{"source": "...", ...}] # hybrid only
    }

Only `type`, `rule_id` and `detail` are always present; empty optionals are
dropped so the JSON stays small.
"""

from __future__ import annotations

from typing import Any

EvidenceType = str  # "rule" | "statistic" | "hybrid" | "profile" | "fallback"


def _base(
    ev_type: EvidenceType,
    rule_id: str,
    detail: str,
    *,
    stage: int | None,
    inputs: dict[str, Any] | None,
    quote: str | None,
) -> dict[str, Any]:
    ev: dict[str, Any] = {"type": ev_type, "rule_id": rule_id, "detail": detail}
    if stage is not None:
        ev["stage"] = stage
    if inputs:
        # Keep values JSON-friendly and compact — these render as chips in the UI.
        ev["inputs"] = {k: v for k, v in inputs.items() if v is not None}
    if quote:
        ev["quote"] = quote[:300]
    return ev


def rule(
    rule_id: str,
    detail: str,
    *,
    stage: int | None = None,
    inputs: dict[str, Any] | None = None,
    quote: str | None = None,
) -> dict[str, Any]:
    """A deterministic rule fired: a lookup, a pattern match, a flag check."""
    return _base("rule", rule_id, detail, stage=stage, inputs=inputs, quote=quote)


def statistic(
    rule_id: str,
    detail: str,
    *,
    measure: str,
    observed: Any,
    threshold: Any = None,
    stage: int | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A measured quantity crossed (or failed to cross) a threshold."""
    ev = _base("statistic", rule_id, detail, stage=stage, inputs=inputs, quote=None)
    ev["measure"] = measure
    ev["observed"] = observed
    if threshold is not None:
        ev["threshold"] = threshold
    return ev


def hybrid(
    rule_id: str,
    detail: str,
    *,
    components: list[dict[str, Any]],
    stage: int | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Several signals combined. Each component is `{source, detail?}` — the
    UI chips them so a reader can see which inputs voted."""
    ev = _base("hybrid", rule_id, detail, stage=stage, inputs=inputs, quote=None)
    ev["components"] = components
    return ev


def profile(
    rule_id: str,
    detail: str,
    *,
    field: str,
    quote: str | None = None,
    stage: int | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read off a Parallel.ai tenant-profile artifact. `field` is the dotted
    path inside the envelope so an operator can go verify it."""
    merged = {"profile_field": field}
    if inputs:
        merged.update(inputs)
    return _base("profile", rule_id, detail, stage=stage, inputs=merged, quote=quote)


def fallback(
    rule_id: str,
    detail: str,
    *,
    stage: int | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """No signal matched — this is the default branch. Distinct from `rule` on
    purpose: "we defaulted" is materially different provenance from "a rule
    fired", and consumers routinely want to audit exactly these."""
    return _base("fallback", rule_id, detail, stage=stage, inputs=inputs, quote=None)


def content_message(dimension: str, *, stage: int | None = None) -> dict[str, Any]:
    """The one explanation every question tagger shares: this element is a
    content message, not a question, so the dimension does not apply.

    Question taggers return `status="skipped"` in this case and `assembly.py`
    drops skipped tags from the output — but the evidence is still built so the
    reason is available to anything reading the accumulator directly.
    """
    return rule(
        f"question.{dimension}.content_message",
        "This element is a content message — instructions, a section header, a "
        "thank-you page — not a question, so there is nothing to classify on this "
        "dimension.",
        stage=stage,
        inputs={"is_content_message": True},
    )


def component(source: str, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    """One entry in a `hybrid(...)` components list."""
    c: dict[str, Any] = {"source": source}
    if detail:
        c["detail"] = detail
    c.update({k: v for k, v in extra.items() if v is not None})
    return c


def detail_of(evidence: str | dict | None) -> str:
    """Read the human sentence out of either evidence shape.

    Taggers that chain off another tag's evidence (and tests) should use this
    rather than assuming a string — `evidence` has been `str | dict` since
    typed evidence landed.
    """
    if isinstance(evidence, dict):
        return str(evidence.get("detail") or "")
    return evidence or ""
