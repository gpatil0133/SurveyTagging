"""Data sensitivity tagger: PII and sensitive data detection via keyword matching."""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers import _sub_types
from taggers.base import QuestionTagger

# Compiled regex patterns for each sensitivity category
_PATTERNS: dict[str, list[re.Pattern]] = {
    "PII \u2013 Name": [
        re.compile(r"\bfull\s*name\b", re.I),
        re.compile(r"\byour\s*name\b", re.I),
        re.compile(r"\bfirst\s*name\b", re.I),
        re.compile(r"\blast\s*name\b", re.I),
        re.compile(r"\benter.*name\b", re.I),
    ],
    "PII \u2013 Email": [
        re.compile(r"\bemail\b", re.I),
        re.compile(r"\be-mail\b", re.I),
    ],
    "PII \u2013 Phone": [
        re.compile(r"\bphone\b", re.I),
        re.compile(r"\bmobile\b", re.I),
        re.compile(r"\btelephone\b", re.I),
        re.compile(r"\bcontact\s*number\b", re.I),
    ],
    "PII \u2013 Location": [
        re.compile(r"\b(?:street|mailing|home|postal)\s*address\b", re.I),
        re.compile(r"\bzip\s*code\b", re.I),
        re.compile(r"\bpostal\s*code\b", re.I),
        re.compile(r"\bcity\s*/?\s*town\b", re.I),
        re.compile(r"^city$", re.I),
        re.compile(r"^address$", re.I),
    ],
    "PII \u2013 ID": [
        re.compile(r"\bcustomer\s*id\b", re.I),
        re.compile(r"\bemployee\s*id\b", re.I),
        re.compile(r"\bpatient\s*id\b", re.I),
        re.compile(r"\bstudent\s*id\b", re.I),
        re.compile(r"\baccount\s*number\b", re.I),
        re.compile(r"\bssn\b", re.I),
        re.compile(r"\bsocial\s*security\b", re.I),
    ],
    "Sensitive \u2013 Health": [
        re.compile(r"\bdiagnosis\b", re.I),
        re.compile(r"\bmedication\b", re.I),
        re.compile(r"\bmental\s*health\b", re.I),
        re.compile(r"\bmedical\s*condition\b", re.I),
        re.compile(r"\bdisability\b", re.I),
        re.compile(r"\bhealth\s*history\b", re.I),
    ],
    "Sensitive \u2013 Financial": [
        re.compile(r"\b(?:household\s*)?income\b", re.I),
        re.compile(r"\bsalary\b", re.I),
        re.compile(r"\bfinancial\s*data\b", re.I),
        re.compile(r"\baccount\s*balance\b", re.I),
    ],
}


class DataSensitivityTagger(QuestionTagger):
    name = "question.data_sensitivity"
    tag_dimension = "data_sensitivity"
    stage = 3
    source_type = "deterministic"
    # This tagger TAGS content messages rather than skipping them, so the
    # base class must not short-circuit them away before it is called.
    skips_content_messages = False

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(
                value=["Anonymous-safe"], source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.data_sensitivity.content_message",
                    "A content message collects no answer at all, so it cannot carry "
                    "personal or sensitive data. Note this dimension returns "
                    "Anonymous-safe rather than skipping, because downstream "
                    "anonymization logic reads every element.",
                    stage=3,
                    inputs={"is_content_message": True},
                ),
            )

        matches: list[str] = []
        # Multi-label: remember what fired for each category, so the evidence can
        # name the trigger instead of just restating the categories.
        why: list[dict] = []

        # Check question title
        title = question.title

        # Check matrix group title as well
        group_title = question.matrix_group_title
        check_text = f"{title} {group_title}"

        # Special subType checks. The code lives in `taggers/_sub_types.py` so
        # this tagger, `role_intent` and the capability layer's
        # `_response_format` cannot disagree about what an email field is.
        if question.question_sub_type in _sub_types.EMAIL:
            matches.append("PII \u2013 Email")

        # Regex pattern matching
        for category, patterns in _PATTERNS.items():
            if category in matches:
                continue
            for pattern in patterns:
                hit = pattern.search(check_text)
                if hit:
                    matches.append(category)
                    why.append(ev.component(
                        category,
                        f'question text matches {pattern.pattern!r} '
                        f'(matched "{hit.group(0)}")',
                    ))
                    break

        # Income detection from answer options
        if not any("Financial" in m for m in matches):
            opt_text = " ".join(o.answer_text.lower() for o in question.answer_options)
            if re.search(r"\$\d+[,.]?\d*", opt_text) and "income" in check_text.lower():
                matches.append("Sensitive \u2013 Financial")
                why.append(ev.component(
                    "Sensitive \u2013 Financial",
                    "the question mentions income and its answer options are currency "
                    "bands \u2014 an income bracket, even though no keyword matched",
                ))

        if not matches:
            return TagResult(
                value=["Anonymous-safe"],
                source="deterministic",
                confidence=1.0,
                evidence=ev.rule(
                    "question.data_sensitivity.no_pii_signal",
                    "Neither the question text nor the matrix group title matched any "
                    "PII or sensitive-data pattern (name, email, phone, address, ID, "
                    "health, financial), and the answer options are not currency "
                    "bands. Safe to report unaggregated.",
                    stage=3,
                    inputs={"question_sub_type": question.question_sub_type},
                    quote=title,
                ),
            )

        return TagResult(
            value=matches,
            source="deterministic",
            confidence=0.95,
            evidence=ev.hybrid(
                "question.data_sensitivity.pattern_match",
                f"The question collects {len(matches)} kind(s) of personal or sensitive "
                "data. Each entry below names the category and the pattern that "
                "detected it \u2014 confidence is 0.95 rather than 1.0 because keyword "
                "matching can misread context.",
                components=why,
                stage=3,
                inputs={"question_sub_type": question.question_sub_type},
            ),
        )


def create_tagger() -> DataSensitivityTagger:
    return DataSensitivityTagger()
