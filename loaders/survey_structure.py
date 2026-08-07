"""Load survey structure from survey_structure.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import sharefs
from loaders.text_cleaner import clean_text, clean_answer_text
from models.survey import AnswerOption, QuestionContext, SurveyMeta

logger = logging.getLogger(__name__)


def _parse_bool_str(val: str | bool | None) -> bool:
    """Convert string 'true'/'false' to Python bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return False


def load_survey_structure(
    survey_dir: Path,
) -> tuple[SurveyMeta, list[QuestionContext]]:
    """Load and parse survey_structure.json from disk.

    Args:
        survey_dir: Path to the survey folder (e.g., .../75885/SurveyData/3/)

    Returns:
        Tuple of (SurveyMeta, list of QuestionContext).
        Returns empty list if questionData is null/missing.
    """
    structure_file = survey_dir / "survey_structure.json"

    if not sharefs.exists(structure_file):
        raise FileNotFoundError(f"survey_structure.json not found in {survey_dir}")

    with sharefs.open_file(structure_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    return parse_survey_data(data)


def parse_survey_data(
    data: dict,
    source_label: str = "input",
) -> tuple[SurveyMeta, list[QuestionContext]]:
    """Parse survey structure from a dict (file-loaded or API-provided).

    Accepts both formats:
      - {"SurveyData": [{...}]}  (standard file format)
      - {top-level survey object}  (convenience: raw survey object)

    Returns:
        Tuple of (SurveyMeta, list of QuestionContext).
    """
    # Handle both wrapper and unwrapped formats
    survey_data = data.get("SurveyData")
    if survey_data and isinstance(survey_data, list) and len(survey_data) > 0:
        raw = survey_data[0]
    elif "questionData" in data or "surveyTitle" in data or "zarcaID" in data:
        raw = data  # Already unwrapped
    else:
        raise ValueError(f"Unrecognized survey structure format from {source_label}")

    # Parse survey-level fields
    title_cleaned = clean_text(raw.get("surveyTitle"))
    themes_raw = raw.get("themes")
    emotions_raw = raw.get("emotions")

    survey_meta = SurveyMeta(
        zarca_id=int(raw.get("zarcaID", 0)),
        corporate_no=int(raw.get("corporateNo", 0)),
        survey_no=int(raw.get("surveyNo", 0)),
        title=title_cleaned.cleaned,
        title_raw=title_cleaned.raw,
        survey_type=str(raw.get("surveyType", "") or ""),
        description=raw.get("surveyDescription"),
        start_date=raw.get("startDate"),
        end_date=raw.get("endDate"),
        sentiment=raw.get("sentiment"),
        themes=_split_csv(themes_raw),
        emotions=_split_csv(emotions_raw),
    )

    # Parse questions
    question_data = raw.get("questionData")
    if not question_data:
        logger.warning("no_question_data", extra={"source": source_label})
        return survey_meta, []

    questions: list[QuestionContext] = []
    for idx, q_raw in enumerate(question_data):
        title_result = clean_text(q_raw.get("questionTitle"))
        q_type = str(q_raw.get("questionType", "") or "")

        options = []
        for opt in q_raw.get("answerOptions", []):
            options.append(AnswerOption(
                answer_id=int(opt.get("answerID", 0)),
                answer_text=clean_answer_text(opt.get("answerText")),
                weight=opt.get("weight"),
            ))

        question = QuestionContext(
            question_id=int(q_raw.get("questionID", 0)),
            question_no=int(q_raw.get("questionNo", 0)),
            position_index=idx,
            title=title_result.cleaned,
            title_raw=title_result.raw,
            question_type=q_type,
            question_sub_type=int(q_raw.get("questionSubType", 0) or 0),
            rs_type=int(q_raw.get("rsType", 0) or 0),
            is_multi=bool(q_raw.get("isMulti", False)),
            matrix_group_title=clean_text(q_raw.get("matrixgrouptitle", "")).cleaned,
            is_custom_metric=_parse_bool_str(q_raw.get("isCustomMetric")),
            custom_metric_title=str(q_raw.get("customMetricTitle", "") or ""),
            calculation_type=str(q_raw.get("calculationType", "") or ""),
            is_followup_question=_parse_bool_str(q_raw.get("isFollowupQuestion")),
            metric_question_id=int(q_raw.get("metricQuestion", 0) or 0),
            is_key_driver=_parse_bool_str(q_raw.get("isKeyDriver")),
            answer_options=options,
            has_piping_markers=title_result.has_piping,
            piping_markers=title_result.piping_markers,
            is_content_message=(q_type == "CM"),
        )
        questions.append(question)

    return survey_meta, questions


def _split_csv(raw: str | None) -> list[str]:
    """Split a comma-separated string into a list of stripped, non-empty values."""
    if not raw:
        return []
    return [v.strip() for v in raw.split(",") if v.strip()]
