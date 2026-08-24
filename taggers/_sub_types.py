"""`questionSubType` code groups for text (`T`) questions.

`questionSubType` on the survey-structure API narrows what a `T` question
actually collects: a date picker, an email-validated field, a file upload, or
plain prose. Three codes are established from live payloads and were already
hard-coded in `role_intent` and `data_sensitivity`; this module is the single
place they are named, so those taggers and
`dashboard_capability::_response_format` cannot drift apart about what a text
question is.

This is NOT the same enum as RMX's `qSubType` from `/AIAllQuestions`, which uses
1-5 for date / time / datetime / date-range / statistical. Same concept,
different numbering — never copy a value between the two.

`NUMERIC` is deliberately EMPTY, and that is the V8 migration's Phase 0
prerequisite showing through. Our numeric/statistical sub-type code has not been
enumerated from live payloads, and guessing it would mis-shape every text
question that happens to carry whatever number we picked. `Numeric-Open` — and
through it `scale_of_measurement: Ratio` and `calculation_type: Sum`, the two
values Phase 1 exists to bring back to life — is therefore fully declared and
wired but unreachable until a real code is added here. Everything downstream
keys off this set, so turning it on is a one-line change followed by an eval
run.
"""

from __future__ import annotations

# Sub-type 1: the platform's date-picker text field.
DATE: frozenset[int] = frozenset({1})

# Sub-type 31: the email-validated text field. Capability-wise identical to a
# CS contact block — an identifier, table-only, nothing to plot.
EMAIL: frozenset[int] = frozenset({31})

# Sub-type 71: file upload. There is no analyzable answer at all; before V8 it
# became a word cloud of filenames.
FILE_UPLOAD: frozenset[int] = frozenset({71})

# Phase 0, unresolved. See the module docstring before filling this in.
NUMERIC: frozenset[int] = frozenset()


def is_plain_text(sub_type: int) -> bool:
    """True when a `T` question is genuine free prose.

    The complement of every recognized sub-type, so a code we have never seen
    reads as prose rather than silently acquiring a shape. That is the safe
    direction: prose is what a `T` question was treated as before V8.
    """
    return sub_type not in (DATE | EMAIL | FILE_UPLOAD | NUMERIC)
