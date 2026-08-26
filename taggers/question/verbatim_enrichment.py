"""verbatim_enrichment tagger — which text-analytics outputs exist for this question.

Stage 3, deterministic. No tag dependencies.

The platform already runs sentiment, theme, emotion and topic extraction over
open-text answers and writes the result to `survey_response_data.parquet`. Nothing
read that file, so every open-text question advertised the same three widgets
("Word Cloud", "Sentiment", "Table") whether its verbatims had been analyzed or
not — and the analytics that WERE there could not be found by a consumer without
opening the parquet itself.

This dimension states what is actually present, per question, so a dashboard can
offer a theme or emotion widget only where there is something to render.
`widget_compatibility` reads it for exactly that (see dashboard_capability.py).

Counts, not content: the evidence carries how many rows carry each enrichment and
the most frequent label names, never a respondent's words.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# Coverage below this reads as "a handful of rows happened to be analyzed" rather
# than a usable basis for a widget. It is recorded either way — the value lists
# what exists — but it lowers confidence so a consumer can tell the difference.
_THIN_COVERAGE = 0.25


class VerbatimEnrichmentTagger(QuestionTagger):
    name = "question.verbatim_enrichment"
    tag_dimension = "verbatim_enrichment"
    stage = 3
    source_type = "deterministic"
    skip_value = []  # multi-label: an empty list, never None

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        signals = context.verbatim_for(question.question_id)

        if signals is None or not signals.n_analyzed:
            # Two different nothings, worth telling apart: a question that takes no
            # free text has none to analyze, while an open-text question with no
            # rows means the analytics have not run (or nobody answered).
            is_text = question.question_type == "T"
            return TagResult(
                value=[], source="deterministic", status="skipped", confidence=1.0,
                evidence=ev.rule(
                    "question.verbatim_enrichment.no_verbatims"
                    if is_text else "question.verbatim_enrichment.not_open_text",
                    "No analyzed verbatims for this question in "
                    "survey_response_data.parquet. The question is open-text, so this "
                    "means the platform's text analytics have not produced anything "
                    "for it yet — re-tagging after they run will populate it."
                    if is_text else
                    f"Question type {question.question_type} collects no free text, so "
                    "there is nothing for text analytics to enrich.",
                    stage=3,
                    inputs={"question_type": question.question_type,
                            "analyzed_rows": 0},
                ),
            )

        available = signals.available
        if not available:
            return TagResult(
                value=[], source="deterministic", status="skipped", confidence=1.0,
                evidence=ev.statistic(
                    "question.verbatim_enrichment.rows_but_no_enrichment",
                    f"{signals.n_analyzed} verbatim row(s) exist for this question but "
                    "none carries sentiment, themes, emotions, topics or an action "
                    "plan. The answers are stored; the analytics have not scored them.",
                    measure="analyzed_rows",
                    observed=signals.n_analyzed,
                    threshold=1,
                    stage=3,
                ),
            )

        # Coverage of the richest signal present, against the rows for this
        # question. Themes and sentiment are usually scored together; taking the
        # max avoids calling a fully-scored question thin because action plans
        # (which only exist for some answers) are sparse.
        best = max(signals.n_sentiment, signals.n_themes,
                   signals.n_emotions, signals.n_topics)
        coverage = best / signals.n_analyzed if signals.n_analyzed else 0.0
        thin = coverage < _THIN_COVERAGE

        return TagResult(
            value=available,
            source="deterministic",
            confidence=0.70 if thin else 1.0,
            evidence=ev.statistic(
                "question.verbatim_enrichment.analytics_present",
                f"{signals.n_analyzed} analyzed verbatim(s): "
                f"{signals.n_sentiment} with sentiment, {signals.n_themes} with themes, "
                f"{signals.n_emotions} with emotions, {signals.n_topics} with topics, "
                f"{signals.n_action_plans} with an action plan. "
                + ("Coverage is thin, so the widgets this enables will be built on a "
                   "fraction of the answers — hence the lower confidence."
                   if thin else
                   "Coverage is good enough for a theme or sentiment widget to be "
                   "worth showing."),
                measure="enrichment_coverage",
                observed=round(coverage, 3),
                threshold=_THIN_COVERAGE,
                stage=3,
                inputs={"analyzed_rows": signals.n_analyzed,
                        "with_sentiment": signals.n_sentiment,
                        "with_themes": signals.n_themes,
                        "with_emotions": signals.n_emotions,
                        "with_topics": signals.n_topics,
                        "with_action_plans": signals.n_action_plans,
                        "top_themes": signals.top_themes,
                        "top_emotions": signals.top_emotions,
                        "top_topics": signals.top_topics},
            ),
        )


def create_tagger() -> VerbatimEnrichmentTagger:
    return VerbatimEnrichmentTagger()
