"""Flow reusability tagger: hybrid scale-fingerprint + LLM."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger

# Known benchmark fingerprint patterns
_BENCHMARK_FINGERPRINTS = {
    "11:0-10:likelihood": "NPS",
    "11:0-10:unlabeled": "NPS (unlabeled)",
    "5:1-5:satisfaction": "CSAT",
    "7:1-7:agreement": "Likert Agreement",
    "5:1-5:effort": "CES",
    "5:1-5:quality": "Quality Scale",
}

# Patterns suggesting custom/one-off
_CUSTOM_INDICATORS = [
    "check mark", "dollar sign", "heart", "star", "thumb",  # emoji/icon scales
]


class FlowReusabilityTagger(QuestionTagger):
    name = "question.flow_reusability"
    tag_dimension = "flow_reusability"
    stage = 5
    depends_on = ["question.role_intent"]
    source_type = "hybrid"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped")

        # Check scale fingerprint against known benchmarks
        if question.scale_fingerprint:
            for fp_pattern, scale_name in _BENCHMARK_FINGERPRINTS.items():
                if question.scale_fingerprint == fp_pattern:
                    return TagResult(
                        value="Benchmark Question",
                        source="deterministic",
                        confidence=0.95,
                        evidence=f"Scale fingerprint matches {scale_name}: {fp_pattern}",
                    )

            # Partial match: same option count and weight range
            parts = question.scale_fingerprint.split(":")
            if len(parts) == 3:
                count_weight = f"{parts[0]}:{parts[1]}"
                for fp_pattern in _BENCHMARK_FINGERPRINTS:
                    fp_parts = fp_pattern.split(":")
                    if f"{fp_parts[0]}:{fp_parts[1]}" == count_weight:
                        return TagResult(
                            value="Benchmark Question",
                            source="hybrid",
                            confidence=0.80,
                            evidence=f"Scale structure matches benchmark ({count_weight})",
                        )

        # Custom/One-off: emoji or icon-based answers
        opt_text = " ".join(o.answer_text.lower() for o in question.answer_options)
        if any(indicator in opt_text for indicator in _CUSTOM_INDICATORS):
            return TagResult(
                value="Custom / One-off",
                source="deterministic",
                confidence=0.90,
                evidence="Emoji/icon-based answer options",
            )

        # Custom/One-off: answers contain specific brand/product names
        role = accumulator.get_question_tag_value(question.question_id, "role_intent")
        if role == "Segmentation" and question.question_type in ("L", "R", "C"):
            # Highly specific options = custom
            return TagResult(
                value="Custom / One-off",
                source="hybrid",
                confidence=0.70,
                evidence="Segmentation question with domain-specific options",
            )

        # Template Question: standard demographic questions
        if role == "Profiling / Demographic":
            return TagResult(
                value="Template Question",
                source="hybrid",
                confidence=0.75,
                evidence="Standard demographic question",
            )

        # Default — requires LLM to determine Candidate for Library
        return TagResult(
            value="Custom / One-off",
            source="llm",
            confidence=0.50,
            evidence="Requires LLM classification",
        )


def create_tagger() -> FlowReusabilityTagger:
    return FlowReusabilityTagger()
