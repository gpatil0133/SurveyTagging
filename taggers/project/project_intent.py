"""project_intent tagger — a one-line statement of what THIS survey asks about.

Stage 4, LLM-owned. `project_purpose` answers "which of eight business
objectives is this?"; this answers "what is it actually about?" at the
granularity a person uses out loud — "Branch visit feedback", "Post-discharge
care follow-up". The taxonomy value is free text (user_defined) because no enum
can hold "branch visit".

The deterministic half of this tagger is only a floor: the survey title with its
boilerplate stripped, held at 0.45 so LLM Call 1 always wins the merge. That
floor is worth having — the title is the customer's own words, and a survey
whose LLM call is skipped or fails still says something useful instead of
nothing.
"""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

# Words that describe the instrument rather than its subject: "Concept Test
# Survey" -> "Concept Test". Stripped off the END freely, but off the FRONT only
# when punctuation shows it was a label ("Survey: Branch Visit", "Survey -
# Branch Visit") — a bare leading one is usually the subject itself, as in
# "Survey Design Feedback".
_BOILERPLATE = (
    "survey", "surveys", "questionnaire", "questionaire", "form", "poll",
    "feedback form", "response form", "template", "copy", "final", "v1", "v2",
)

# Edition markers: a year, a quarter, a fiscal year, a month. "Q3 2026 Branch
# Visit Feedback" and "Branch Visit Feedback 2026" are the same intent.
_EDITION = re.compile(
    r"^(?:19|20)\d{2}$|^q[1-4]$|^fy\s?\d{2,4}$|^h[12]$|"
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*$",
    re.I,
)

# Separators the title used to bolt an edition or a client name on.
_TRIM_CHARS = " \t-–—_|:;,./\\•()[]{}\"'"

_MAX_WORDS = 8
_MAX_CHARS = 80


def _bare(token: str) -> str:
    return token.strip(_TRIM_CHARS).lower()


def _strip_edges(tokens: list[str]) -> list[str]:
    """Drop boilerplate and edition tokens from the ends until neither matches.

    Iterative rather than a single pass: "Branch Visit Feedback Survey 2026"
    ends in an edition token *behind* a boilerplate one, and one pass would
    leave "Branch Visit Feedback Survey".
    """
    def drop_trailing(token: str) -> bool:
        bare = _bare(token)
        return not bare or bare in _BOILERPLATE or bool(_EDITION.match(bare))

    def drop_leading(index: int) -> bool:
        raw = tokens[index]
        bare = _bare(raw)
        if not bare or _EDITION.match(bare):
            return True
        if bare not in _BOILERPLATE:
            return False
        # Punctuation on the word ("Survey:") or a separator right after it
        # ("Survey - Branch Visit") marks it as a label rather than the subject.
        next_is_separator = len(tokens) > index + 1 and not _bare(tokens[index + 1])
        return raw != raw.strip(_TRIM_CHARS) or next_is_separator

    changed = True
    while tokens and changed:
        changed = False
        if drop_leading(0):
            tokens.pop(0)
            changed = True
        if tokens and drop_trailing(tokens[-1]):
            tokens.pop()
            changed = True
    return tokens


def intent_from_title(title: str) -> str:
    """The survey title reduced to its subject, or "" when nothing survives.

    Public because it is the one piece of this tagger worth testing directly,
    and because the same normalization is what makes the seed comparable across
    a tenant's yearly re-runs of the same survey.
    """
    cleaned = " ".join((title or "").split())
    if not cleaned:
        return ""
    tokens = _strip_edges(cleaned.split(" "))
    if not tokens:
        return ""
    intent = " ".join(tokens[:_MAX_WORDS]).strip(_TRIM_CHARS)
    return intent[:_MAX_CHARS].strip(_TRIM_CHARS)


class ProjectIntentTagger(ProjectTagger):
    name = "project.project_intent"
    tag_dimension = "project_intent"
    stage = 4
    source_type = "llm"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        title = context.survey_meta.title or ""
        seed = intent_from_title(title)

        if not seed:
            # No title, or a title that was nothing but boilerplate ("2026
            # Survey"). pending_llm rather than a guess: the questions carry the
            # answer and only LLM Call 1 reads them.
            return TagResult(
                value=None, source="llm", confidence=0.0, status="pending_llm",
                evidence=ev.fallback(
                    "project.project_intent.no_usable_title",
                    f"The survey title ({title.strip() or 'empty'}) is boilerplate or "
                    "missing, so there is nothing to seed an intent from. LLM Call 1 "
                    "reads the questions and writes it; if you are reading this in the "
                    "output, that call did not run or did not answer.",
                    stage=4,
                    inputs={"survey_title": title.strip() or "(empty)"},
                ),
            )

        return TagResult(
            value=seed,
            source="hybrid",
            # Deliberately below the 0.80 floor that protects a deterministic tag
            # from LLM Call 1: the title names the instrument, the questions say
            # what it asks, and the model is the only thing here that reads them.
            confidence=0.45,
            evidence=ev.rule(
                "project.project_intent.title_stripped",
                "Seeded from the survey title with the instrument words and the "
                f'edition markers removed, leaving "{seed}". A floor, not a finding — '
                "LLM Call 1 rewrites it from the questions and section headers, which "
                "is where a title like \"Q3 CX Survey\" actually gets its meaning.",
                stage=4,
                inputs={"stripped_to": seed, "word_count": len(seed.split())},
                quote=title,
            ),
        )


def create_tagger() -> ProjectIntentTagger:
    return ProjectIntentTagger()
