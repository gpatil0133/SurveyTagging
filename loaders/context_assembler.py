"""Assemble UnifiedContext from all data sources for a single survey."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import sharefs
from loaders.directory import load_directory_signals
from loaders.invitations import load_invitation_signals
from loaders.responses import load_response_stats
from loaders.survey_structure import load_survey_structure, parse_survey_data
from models.context import ExistingTags, MatrixGroup, UnifiedContext
from models.overrides import ManualOverrides
from models.signals import DirectorySignals
from models.survey import QuestionContext
from models.tenant_profile import TenantProfile

logger = logging.getLogger(__name__)


def _compute_section_for_qid(questions: list[QuestionContext]) -> dict[int, str]:
    """Single forward walk: each non-CM question is mapped to the title of the
    most recent preceding `is_content_message=True` question. CM questions
    map to empty string. Used by V5 question-signature builder."""
    out: dict[int, str] = {}
    last_section = ""
    for q in questions:
        if q.is_content_message:
            if q.title:
                last_section = q.title
            out[q.question_id] = ""
        else:
            out[q.question_id] = last_section
    return out


def assemble_context_from_json(
    survey_json: dict,
    manual_overrides: dict | None = None,
) -> UnifiedContext:
    """Build a UnifiedContext from raw JSON (for API / web UI use).

    No filesystem access — works entirely from in-memory data.

    Args:
        survey_json: Raw survey structure dict (with or without SurveyData wrapper).
        manual_overrides: Optional dict with keys industry, company_name,
            department, purpose, country.

    Returns:
        UnifiedContext ready for tagging.
    """
    survey_meta, questions = parse_survey_data(survey_json, source_label="api")
    logger.debug(
        "assemble_context_from_json",
        extra={"survey_no": survey_meta.survey_no,
               "survey_name": survey_meta.title,
               "questions": len(questions),
               "has_overrides": bool(manual_overrides)},
    )

    raw = manual_overrides or {}
    overrides = ManualOverrides(
        industry=str(raw.get("industry", "") or ""),
        company_name=str(raw.get("company_name", "") or ""),
        department=str(raw.get("department", "") or ""),
        purpose=str(raw.get("purpose", "") or ""),
        country=str(raw.get("country", "") or ""),
    )

    # Compute derived fields
    _compute_effective_positions(questions)
    question_groups = _compute_matrix_groups(questions)
    _apply_matrix_group_sizes(questions, question_groups)
    piping_map = _compute_piping_map(questions)
    _compute_scale_fingerprints(questions)

    existing_tags = None
    if survey_meta.sentiment or survey_meta.themes or survey_meta.emotions:
        existing_tags = ExistingTags(
            sentiment=survey_meta.sentiment,
            themes=survey_meta.themes,
            emotions=survey_meta.emotions,
        )

    return UnifiedContext(
        tenant_id=survey_meta.corporate_no,
        overrides=overrides,
        survey_meta=survey_meta,
        questions=questions,
        response_stats=None,
        directory_signals=DirectorySignals(),
        invitation_signals=None,
        has_linking=False,
        has_prepop=False,
        question_groups=question_groups,
        piping_map=piping_map,
        existing_tags=existing_tags,
        section_for_qid=_compute_section_for_qid(questions),
    )


def assemble_context(
    tenant_dir: Path,
    survey_no: int,
    tenant_id: int,
    directory_signals: DirectorySignals | None = None,
    config_dir: Path | None = None,
    tenant_profile: TenantProfile | None = None,
    tenant_canon=None,             # V5: TenantCanon | None  (CX)
    canon_embeddings=None,         # V5: CanonEmbeddingIndex | None  (CX)
    tenant_canon_ex=None,          # V5: TenantCanon | None  (EX)
    canon_embeddings_ex=None,      # V5: CanonEmbeddingIndex | None  (EX)
) -> UnifiedContext:
    """Build the complete UnifiedContext for a single survey.

    Args:
        tenant_dir: Path to the tenant folder.
        survey_no: Survey number within the tenant.
        tenant_id: Numeric tenant ID.
        directory_signals: Pre-loaded directory signals (cached per tenant).
        config_dir: Path to config/ for domain keyword definitions.
        tenant_profile: Pre-loaded TenantProfile (cached per tenant).
            Optional — None means no Parallel.ai artifacts are available.
        tenant_canon: Pre-loaded CX TenantCanon (V5; cached per tenant).
        canon_embeddings: Pre-loaded CX CanonEmbeddingIndex (V5; cached per tenant).
        tenant_canon_ex: Pre-loaded EX TenantCanon (V5; cached per tenant).
        canon_embeddings_ex: Pre-loaded EX CanonEmbeddingIndex (V5; cached per tenant).

    Returns:
        Fully assembled UnifiedContext ready for tagging.
    """
    survey_dir = tenant_dir / "SurveyData" / str(survey_no)
    logger.debug("assemble_context_start",
                 extra={"tenant": tenant_id, "survey": survey_no,
                        "survey_dir": str(survey_dir)})

    # Load survey structure
    survey_meta, questions = load_survey_structure(survey_dir)
    logger.debug("survey_structure_loaded",
                 extra={"survey": survey_no, "survey_name": survey_meta.title,
                        "questions": len(questions)})

    # Load response stats
    response_stats = load_response_stats(survey_dir)

    # Load directory signals (use cached if provided)
    if directory_signals is None:
        directory_signals = load_directory_signals(tenant_dir, config_dir)

    # Load invitation signals
    invitation_signals = load_invitation_signals(survey_dir)

    # Check existence of linking and prepop files
    has_linking = sharefs.exists(survey_dir / "directory_linking.parquet")
    has_prepop = sharefs.exists(survey_dir / "prepop_data.parquet")

    # Compute derived fields
    _compute_effective_positions(questions)
    question_groups = _compute_matrix_groups(questions)
    _apply_matrix_group_sizes(questions, question_groups)
    piping_map = _compute_piping_map(questions)
    _compute_scale_fingerprints(questions)
    logger.debug(
        "context_derived_fields_computed",
        extra={"survey": survey_no, "matrix_groups": len(question_groups),
               "piped_questions": len(piping_map),
               "has_responses": response_stats is not None,
               "has_invitations": invitation_signals is not None,
               "has_linking": has_linking, "has_prepop": has_prepop},
    )

    # Extract existing tags
    existing_tags = None
    if survey_meta.sentiment or survey_meta.themes or survey_meta.emotions:
        existing_tags = ExistingTags(
            sentiment=survey_meta.sentiment,
            themes=survey_meta.themes,
            emotions=survey_meta.emotions,
        )

    return UnifiedContext(
        tenant_id=tenant_id,
        tenant_profile=tenant_profile,
        tenant_canon=tenant_canon,
        canon_embeddings=canon_embeddings,
        tenant_canon_ex=tenant_canon_ex,
        canon_embeddings_ex=canon_embeddings_ex,
        survey_meta=survey_meta,
        questions=questions,
        response_stats=response_stats,
        directory_signals=directory_signals,
        invitation_signals=invitation_signals,
        has_linking=has_linking,
        has_prepop=has_prepop,
        question_groups=question_groups,
        piping_map=piping_map,
        existing_tags=existing_tags,
        section_for_qid=_compute_section_for_qid(questions),
    )


def _compute_effective_positions(questions: list[QuestionContext]) -> None:
    """Compute position ratio for each question among non-CM questions."""
    non_cm = [q for q in questions if not q.is_content_message]
    total = len(non_cm)

    if total == 0:
        return

    for idx, q in enumerate(non_cm):
        if total == 1:
            q.effective_position_ratio = 0.0
        else:
            q.effective_position_ratio = idx / (total - 1)


def _compute_matrix_groups(questions: list[QuestionContext]) -> list[MatrixGroup]:
    """Identify matrix groups: questions sharing the same questionNo + matrixgrouptitle."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)

    for q in questions:
        if q.matrix_group_title and not q.is_content_message:
            key = (q.question_no, q.matrix_group_title)
            groups[key].append(q.question_id)

    result = []
    for (q_no, title), q_ids in groups.items():
        if len(q_ids) > 1:
            result.append(MatrixGroup(
                question_no=q_no,
                group_title=title,
                question_ids=q_ids,
                size=len(q_ids),
            ))

    return result


def _apply_matrix_group_sizes(
    questions: list[QuestionContext],
    groups: list[MatrixGroup],
) -> None:
    """Set matrix_group_size on each question that belongs to a group."""
    id_to_size: dict[int, int] = {}
    for group in groups:
        for q_id in group.question_ids:
            id_to_size[q_id] = group.size

    for q in questions:
        if q.question_id in id_to_size:
            q.matrix_group_size = id_to_size[q.question_id]


def _compute_piping_map(questions: list[QuestionContext]) -> dict[int, list[str]]:
    """Build a map of question_id → piping markers found in that question."""
    piping_map: dict[int, list[str]] = {}
    for q in questions:
        if q.piping_markers:
            piping_map[q.question_id] = q.piping_markers
    return piping_map


def _compute_scale_fingerprints(questions: list[QuestionContext]) -> None:
    """Compute a normalized fingerprint for each question's answer scale.

    Fingerprint format: "{option_count}:{min_weight}-{max_weight}:{pattern}"
    """
    for q in questions:
        if q.is_content_message or not q.answer_options:
            continue

        weights = [o.weight for o in q.answer_options if o.weight is not None]
        if not weights:
            continue

        min_w = min(weights)
        max_w = max(weights)
        count = len(q.answer_options)

        # Determine pattern from answer text
        texts = [o.answer_text.lower().strip() for o in q.answer_options if o.answer_text.strip()]

        pattern = "unknown"
        if _has_any_keyword(texts, ["satisfied", "dissatisfied", "satisfaction"]):
            pattern = "satisfaction"
        elif _has_any_keyword(texts, ["likely", "likelihood"]):
            pattern = "likelihood"
        elif _has_any_keyword(texts, ["agree", "disagree", "agreement"]):
            pattern = "agreement"
        elif _has_any_keyword(texts, ["easy", "difficult", "effort"]):
            pattern = "effort"
        elif _has_any_keyword(texts, ["excellent", "poor", "good", "average"]):
            pattern = "quality"
        elif not texts:
            pattern = "unlabeled"

        q.scale_fingerprint = f"{count}:{min_w:.0f}-{max_w:.0f}:{pattern}"


def _has_any_keyword(texts: list[str], keywords: list[str]) -> bool:
    joined = " ".join(texts)
    return any(kw in joined for kw in keywords)
