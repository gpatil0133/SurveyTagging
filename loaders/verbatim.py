"""Load NLP enrichment for open-text answers from `survey_response_data.parquet`.

This file sits in every survey directory and was read by nothing. It holds what
the platform's own text analytics already produced per verbatim answer:

    ResponseUid, FeedbackQuestionId, Qno, ResponseStartDate, FeedbackAnswer,
    Sentiment ("positive"/"neutral"/"negative"), SentimentScore, ConfidenceScore,
    ActionPlan, MetricQuestion, Themes, Emotions, Topics

`Themes`, `Emotions` and `Topics` are JSON arrays stored as strings:

    Themes   [{"Text": "Wait Time", "Sentiment": "neutral",
               "ConfidenceScore": -0.2, "AssociatedPhrases": "visit was fine"}]
    Emotions [{"Text": "Acceptance", "Sentiment": "neutral", "ConfidenceScore": -0.21}]
    Topics   [{"TopicId": 10301, "TopicName": "Banking & Finance"}]

Only the shape and volume of that payload is read here, never the verbatim text
itself: tagging says what a question CAN support, and quoting a respondent's words
into `tagged_output.json` would move personal data into an artifact whose whole
point is to be widely readable. `MetricQuestion` is deliberately ignored — it is
empty in every row on the share (see `metric_question_id` in models/survey.py).

Keyed by `FeedbackQuestionId`, which is the `questionID` of the open-text question
the answers belong to, so a tagger can ask "is THIS question enriched?".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

import sharefs
from models.signals import VerbatimSignals

logger = logging.getLogger(__name__)

RESPONSE_DATA_FILE = "survey_response_data.parquet"

# Only these are read. Naming them keeps the verbatim text (`FeedbackAnswer`) out
# of the process entirely rather than relying on nobody touching it later.
_COLUMNS = ["FeedbackQuestionId", "Sentiment", "SentimentScore", "ActionPlan",
            "Themes", "Emotions", "Topics"]

# How many distinct labels to keep per question. The list is evidence for a
# reviewer ("this is what the analytics found"), not a data feed — the dashboard
# service reads the parquet itself.
_TOP_LABELS = 8


def _labels(raw: str | None, key: str) -> list[str]:
    """Names out of one JSON array cell, or [] for anything unparseable."""
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get(key) or "").strip()
            if name:
                out.append(name)
    return out


def load_verbatim_signals(survey_dir: Path) -> dict[int, VerbatimSignals]:
    """`{question_id: VerbatimSignals}` for every enriched open-text question.

    Empty dict when the file is absent, unreadable, or carries no rows — a survey
    whose verbatims have not been analyzed is the normal case, not an error.
    """
    path = survey_dir / RESPONSE_DATA_FILE
    if not sharefs.exists(path):
        return {}

    try:
        with sharefs.open_file(path, "rb") as fh:
            table = pq.read_table(fh, columns=_COLUMNS)
    except Exception as e:  # noqa: BLE001 — pyarrow raises its own types
        logger.warning("verbatim_data_read_failed",
                       extra={"path": str(path), "error": f"{type(e).__name__}: {e}"})
        return {}

    if table.num_rows == 0:
        return {}

    rows = table.to_pylist()
    per_q: dict[int, dict] = {}

    for row in rows:
        raw_qid = str(row.get("FeedbackQuestionId") or "").strip()
        if not raw_qid.isdigit():
            continue
        acc = per_q.setdefault(int(raw_qid), {
            "n": 0, "sentiment": 0, "scored": 0, "themes": 0, "emotions": 0,
            "topics": 0, "action_plans": 0,
            "theme_names": {}, "emotion_names": {}, "topic_names": {},
        })
        acc["n"] += 1

        if str(row.get("Sentiment") or "").strip():
            acc["sentiment"] += 1
        # Presence, not truthiness: "0.00" is a real score and "" is not.
        if str(row.get("SentimentScore") or "").strip():
            acc["scored"] += 1
        if str(row.get("ActionPlan") or "").strip():
            acc["action_plans"] += 1

        for field, count_key, names_key, name_field in (
            ("Themes", "themes", "theme_names", "Text"),
            ("Emotions", "emotions", "emotion_names", "Text"),
            ("Topics", "topics", "topic_names", "TopicName"),
        ):
            names = _labels(row.get(field), name_field)
            if names:
                acc[count_key] += 1
            for name in names:
                acc[names_key][name] = acc[names_key].get(name, 0) + 1

    def top(counts: dict[str, int]) -> list[str]:
        # Frequency first, then alphabetical so the list is stable run to run —
        # an unstable list would churn `tagged_output.json` on every re-tag.
        return [n for n, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:_TOP_LABELS]

    signals = {
        qid: VerbatimSignals(
            question_id=qid,
            n_analyzed=acc["n"],
            n_sentiment=acc["sentiment"],
            n_sentiment_scored=acc["scored"],
            n_themes=acc["themes"],
            n_emotions=acc["emotions"],
            n_topics=acc["topics"],
            n_action_plans=acc["action_plans"],
            top_themes=top(acc["theme_names"]),
            top_emotions=top(acc["emotion_names"]),
            top_topics=top(acc["topic_names"]),
        )
        for qid, acc in per_q.items()
    }

    logger.debug("verbatim_signals_loaded",
                 extra={"path": str(path), "questions": len(signals),
                        "rows": table.num_rows})
    return signals
