"""widget_footprint tagger — how much dashboard grid space the widget should
occupy (V8, Phase 4).

Stage 5, deterministic. Depends on display_role (Stage 5, and alphabetically
earlier so it has already run) and block_id (Stage 3).

**What it fills.** `cols` and `rows` on the widget payload. Every widget ships
12x6 today, so any other size costs a second API round-trip per widget — the
fixed-geometry limitation recorded in
`docs/custom-dashboard-end-to-end-flow.md` §11. Stating the intended footprint
in the tag is what lets a composer batch that correction, or skip it when 12x6
is already right.

The grid is 24 columns wide, so Half (12) is two per row and Quarter (6) is
four — which is the point of the Primary KPI rule: four score tiles sit in one
row across the top of the dashboard, which is how a KPI band is meant to read.

Deterministic, no LLM cost: this is a layout consequence of `display_role`, and
if a reader disagrees with the footprint the thing to fix is the role.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

QUARTER = "Quarter (6x6)"
HALF = "Half (12x6)"
FULL_WIDTH = "Full-Width (24x6)"

# Declared as part of the grid vocabulary, and deliberately not emitted by any
# rule below. Nothing in the taxonomy tells us a widget needs vertical room
# rather than horizontal — that is a property of how many rows the DATA has,
# which the dashboard service learns when it checks viability and this Stage-5
# tagger never sees. A composer that has counted the rows may still choose it.
TALL = "Tall (12x12)"

# display_role -> footprint. Everything that is not a headline number is Half:
# two per row is the readable default, and it is also the size the platform
# already ships, so those widgets cost no correction round-trip at all.
_BY_DISPLAY_ROLE: dict[str, str] = {
    "Primary KPI": QUARTER,
    "Supporting Metric": HALF,
    "Comparison Metric": HALF,
    "Trend Indicator": HALF,
    "Drill-down": HALF,
    "Detail View": HALF,
}


class WidgetFootprintTagger(QuestionTagger):
    name = "question.widget_footprint"
    tag_dimension = "widget_footprint"
    stage = 5
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.display_role", "question.block_id"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("widget_footprint", stage=5))

        block = accumulator.get_question_tag_value(q.question_id, "block_id")
        role = accumulator.get_question_tag_value(q.question_id, "display_role")

        # A matrix block is rendered as ONE widget covering every row, and a
        # stacked bar with a dozen sentence-length row labels needs the width.
        # Checked before display_role because the footprint follows the widget,
        # and the widget here is the whole block rather than this question.
        if block:
            return TagResult(
                value=FULL_WIDTH, source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.widget_footprint.matrix_block",
                    f'This question is one row of the matrix block "{block}", which the '
                    "dashboard renders as a single widget covering every row. A stacked "
                    "bar with that many sentence-length row labels needs the full 24 "
                    "columns to stay readable.",
                    stage=5,
                    inputs={"block_id": block,
                            "display_role": role or "(unset)",
                            "matrix_group_size": q.matrix_group_size},
                ),
            )

        footprint = _BY_DISPLAY_ROLE.get(str(role or ""))
        if footprint == QUARTER:
            return TagResult(
                value=QUARTER, source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.widget_footprint.headline_tile",
                    "display_role is Primary KPI — a single headline number. At a "
                    "quarter of the 24-column grid, four of them sit in one row across "
                    "the top of the dashboard, which is how a KPI band is meant to "
                    "read. A half-width tile holding one number is mostly empty space.",
                    stage=5,
                    inputs={"display_role": role},
                ),
            )
        if footprint:
            return TagResult(
                value=HALF, source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.widget_footprint.standard_half",
                    f"display_role is {role} — a chart or a list rather than a headline "
                    "number, so it needs a chart's worth of room but not the full "
                    "width. Half the grid puts two side by side, and it is also the "
                    "size the platform already ships, so this widget costs no geometry "
                    "correction at all.",
                    stage=5,
                    inputs={"display_role": role},
                ),
            )

        return TagResult(
            value=HALF, source="deterministic", confidence=0.60,
            evidence=ev.fallback(
                "question.widget_footprint.no_role",
                f"display_role is {role or 'unset'}, so no layout rule applies. Half is "
                "the fallback because it is what the platform ships by default — the "
                "one choice that is never wrong enough to need a second API call to "
                "fix.",
                stage=5,
                inputs={"display_role": role or "(unset)"},
            ),
        )


def create_tagger() -> WidgetFootprintTagger:
    return WidgetFootprintTagger()
