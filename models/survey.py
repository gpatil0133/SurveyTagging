"""Survey structure and question models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerOption(BaseModel):
    """A single answer choice for a question."""

    answer_id: int
    answer_text: str = ""
    weight: float | None = None


class QuestionContext(BaseModel):
    """A single question with all metadata needed for tagging."""

    question_id: int
    question_no: int
    position_index: int = 0  # 0-based position in the full question list

    title: str = ""           # HTML-cleaned, piping markers stripped
    title_raw: str = ""       # Original title with HTML/piping intact

    question_type: str = ""   # CM, T, L, C, R, RT, RS, RW, RK, HR, GR, GC, GQ, SR, RG, ML, CS
    question_sub_type: int = 0
    rs_type: int = 0          # 0=standard, 2=NPS, 3=CES (Customer Effort Score), 4=CSAT-5

    is_multi: bool = False
    matrix_group_title: str = ""
    is_custom_metric: bool = False
    custom_metric_title: str = ""
    calculation_type: str = ""
    is_followup_question: bool = False
    metric_question_id: int = 0  # Parent question ID for follow-ups
    is_key_driver: bool = False

    answer_options: list[AnswerOption] = Field(default_factory=list)

    # Computed during loading
    has_piping_markers: bool = False
    piping_markers: list[str] = Field(default_factory=list)
    is_content_message: bool = False
    effective_position_ratio: float = 0.0  # Position among non-CM questions (0.0 to 1.0)
    matrix_group_size: int = 1             # Number of questions in this matrix group
    scale_fingerprint: str | None = None   # Normalized answer pattern for benchmark detection

    @property
    def is_nps(self) -> bool:
        return self.rs_type == 2

    @property
    def is_csat(self) -> bool:
        return self.rs_type == 4

    @property
    def is_ces(self) -> bool:
        """Customer Effort Score (rs_type=3). Was mis-labeled 'Likert' in v1."""
        return self.rs_type == 3

    @property
    def option_count(self) -> int:
        return len(self.answer_options)


class SurveyMeta(BaseModel):
    """Top-level survey metadata."""

    zarca_id: int
    corporate_no: int
    survey_no: int
    title: str = ""           # HTML-cleaned
    title_raw: str = ""       # Original
    survey_type: str = ""     # Survey, CX, EX, Assessment, Poll
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    # Pre-existing tags (populated for ~4% of surveys)
    sentiment: str | None = None
    themes: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
