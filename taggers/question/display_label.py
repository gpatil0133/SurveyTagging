"""display_label tagger — a short chart title for any chart-eligible question
(V8, Phase 4).

Stage 5, llm-refined, user_defined. Depends on metric_name and
widget_compatibility (both Stage 3).

**What it fills.** `title` on the widget payload. `metric_name` covers only
measured questions and `sub_stage_name` only journey-eligible metrics, so a
plain segmentation question's widget currently gets the raw question text — often
a full sentence, sometimes with the piping markers still in it — as its chart
title.

**Derivation, and why it is shaped this way.** When `metric_name` exists it IS
the label, written at a confidence above the LLM override threshold. That is not
an economy: the two would otherwise be free to disagree, and a dashboard showing
"Overall Satisfaction" in the tag and "How happy were you?" on the chart is
worse than either alone. Only when no metric name exists does LLM Call 2 write a
2-6 word Title Case phrase from the question text, with a cleaned truncation of
the title as the seed — held well below the override threshold so the model
always wins, and left in place as an honest fallback when the call does not land.

Skipped when `widget_compatibility` is empty: a question with no chartable
widget has no chart to title.
"""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# The cap the response parser also enforces. A chart title is one line on a
# tile; anything longer is truncated by the renderer instead, invisibly.
MAX_CHARS = 60
MAX_WORDS = 6

_WHITESPACE_RE = re.compile(r"\s+")
# Leading "Q3.", "12)" and similar numbering the author typed into the title.
_LEADING_NUMBER_RE = re.compile(r"^\s*(?:q\s*)?\d+\s*[.)\-:]\s*", re.I)

# Words that frame a question but carry none of its subject. Dropped from the
# FRONT and the BACK only, never the middle: "How satisfied are you with
# checkout?" must keep "satisfied with checkout" rather than losing the words
# that hold the sentence together. Trimming both ends is what turns "Which
# region are you in?" into "Region" instead of "Region Are You In".
_LEADING_STOPWORDS = {
    "how", "what", "which", "when", "where", "who", "why", "do", "does", "did",
    "is", "are", "was", "were", "would", "will", "have", "has", "can", "could",
    "please", "kindly", "rate", "select", "choose", "tell", "us", "your", "you",
    "the", "a", "an", "to", "on", "in", "of", "at", "for", "and", "or",
    "overall", "much", "many", "likely", "well",
}


def clean_title_to_label(title: str) -> str:
    """A readable chart title truncated out of the question's own wording.

    A seed, not an answer: it keeps the subject and drops the interrogative
    scaffolding, which is enough to beat the raw sentence and not enough to beat
    a model that has read the question.
    """
    if not title:
        return ""
    text = _LEADING_NUMBER_RE.sub("", _WHITESPACE_RE.sub(" ", title).strip())
    text = text.strip().rstrip("?.!:;, ").strip()
    if not text:
        return ""

    words = text.split()
    while words and words[0].lower().strip(",") in _LEADING_STOPWORDS:
        words.pop(0)
    while words and words[-1].lower().strip(",") in _LEADING_STOPWORDS:
        words.pop()
    if not words:
        words = text.split()

    short = words[:MAX_WORDS]
    titled = " ".join(w if w.isupper() else w.capitalize() for w in short)
    return titled[:MAX_CHARS].strip().rstrip(",-")


class DisplayLabelTagger(QuestionTagger):
    name = "question.display_label"
    tag_dimension = "display_label"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_name", "question.widget_compatibility"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("display_label", stage=5))

        widgets = accumulator.get_question_tag_value(q.question_id, "widget_compatibility")
        if not widgets:
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.display_label.no_widget",
                    "widget_compatibility is empty — no chart or table can render this "
                    "question — so there is no widget whose title this would be.",
                    stage=5,
                    inputs={"widget_compatibility": []},
                ),
            )

        metric_name = accumulator.get_question_tag_value(q.question_id, "metric_name")
        if metric_name:
            return TagResult(
                value=str(metric_name)[:MAX_CHARS], source="deterministic",
                confidence=0.90,
                evidence=ev.rule(
                    "question.display_label.from_metric_name",
                    f'The question already has a metric name — "{metric_name}" — and '
                    "that is what the chart should say. Written above the LLM override "
                    "threshold on purpose: if the two were free to disagree, the tag and "
                    "the chart title would name the same measure differently, which is "
                    "worse than either of them alone.",
                    stage=5,
                    inputs={"metric_name": metric_name},
                ),
            )

        seed = clean_title_to_label(q.title)
        if seed:
            return TagResult(
                value=seed, source="hybrid", confidence=0.45,
                evidence=ev.fallback(
                    "question.display_label.title_truncation",
                    "No metric name, so there is nothing authoritative to title the "
                    "chart with. The seed is the question's own wording with the "
                    "numbering and the interrogative opening stripped and the rest "
                    f"Title-Cased ({MAX_WORDS} words max). Held at 0.45 so LLM Call 2 "
                    "always wins — it can read what the question is ABOUT, which "
                    "truncation cannot — and left in place as an honest fallback when "
                    "that call does not land.",
                    stage=5,
                    inputs={"derived_from": "question title",
                            "question_type": q.question_type},
                ),
            )

        return TagResult(
            value=None, source="hybrid", status="pending_llm", confidence=0.0,
            evidence=ev.rule(
                "question.display_label.no_usable_title",
                "The question has no metric name and no title text left after cleaning, "
                "so there is nothing to derive a label from. Handed to LLM Call 2, which "
                "can read the answer options and the section header.",
                stage=5,
                inputs={"question_type": q.question_type, "title": q.title or "(empty)"},
            ),
        )


def create_tagger() -> DisplayLabelTagger:
    return DisplayLabelTagger()
