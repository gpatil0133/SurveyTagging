"""Projections over tagged surveys.

Currently only `survey_view` ships — a unified per-survey response shape.
Per-tenant Custom Journey, per-survey Dashboard, etc. were archived during
the V6.1 cleanup; see BACKLOG.md for the work to bring them back.
"""

from projections.survey_view import build_survey_view

__all__ = ["build_survey_view"]
