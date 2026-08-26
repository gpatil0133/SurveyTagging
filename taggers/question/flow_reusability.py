"""Flow reusability tagger: hybrid scale-fingerprint + LLM."""

from models import evidence as ev
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

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        # Check scale fingerprint against known benchmarks
        if question.scale_fingerprint:
            for fp_pattern, scale_name in _BENCHMARK_FINGERPRINTS.items():
                if question.scale_fingerprint == fp_pattern:
                    return TagResult(
                        value="Benchmark Question",
                        source="deterministic",
                        confidence=0.95,
                        evidence=ev.rule(
                            "question.flow_reusability.exact_benchmark_scale",
                            f"The answer scale is an exact match for the {scale_name} "
                            "fingerprint — same option count, same weight range, same "
                            "label style. A question on a standard scale can be "
                            "compared against outside benchmarks.",
                            stage=5,
                            inputs={"scale_fingerprint": fp_pattern,
                                    "benchmark": scale_name},
                        ),
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
                            evidence=ev.rule(
                                "question.flow_reusability.partial_benchmark_scale",
                                f"The scale has the same shape as a known benchmark "
                                f"({count_weight} — matching option count and weight "
                                "range) but different labelling, so it is probably a "
                                "reworded standard scale rather than an exact one.",
                                stage=5,
                                inputs={"scale_fingerprint":
                                            question.scale_fingerprint,
                                        "matched_structure": count_weight,
                                        "closest_benchmark":
                                            _BENCHMARK_FINGERPRINTS[fp_pattern]},
                            ),
                        )

        # Custom/One-off: emoji or icon-based answers
        opt_text = " ".join(o.answer_text.lower() for o in question.answer_options)
        if any(indicator in opt_text for indicator in _CUSTOM_INDICATORS):
            return TagResult(
                value="Custom / One-off",
                source="deterministic",
                confidence=0.90,
                evidence=ev.rule(
                    "question.flow_reusability.icon_scale",
                    "The answer options are icons or emoji (stars, hearts, thumbs). "
                    "Presentation like this is chosen for one survey's look and does "
                    "not port to a reusable library item.",
                    stage=5,
                    inputs={"matched_indicators":
                                [i for i in _CUSTOM_INDICATORS if i in opt_text]},
                ),
            )

        # Custom/One-off: answers contain specific brand/product names
        role = accumulator.get_question_tag_value(question.question_id, "role_intent")
        if role == "Segmentation" and question.question_type in ("L", "R", "C"):
            # Highly specific options = custom
            return TagResult(
                value="Custom / One-off",
                source="hybrid",
                confidence=0.70,
                evidence=ev.rule(
                    "question.flow_reusability.tenant_specific_segmentation",
                    "A segmentation question with a picklist. Its options name this "
                    "customer's own regions, products or departments, so it would be "
                    "meaningless in another tenant's survey.",
                    stage=5,
                    inputs={"role_intent": "Segmentation",
                            "question_type": question.question_type},
                ),
            )

        # Template Question: standard demographic questions
        if role == "Profiling / Demographic":
            return TagResult(
                value="Template Question",
                source="hybrid",
                confidence=0.75,
                evidence=ev.rule(
                    "question.flow_reusability.demographic_template",
                    "A profiling/demographic question. Age, tenure, region and the "
                    "like are asked the same way everywhere, so this is template "
                    "material even though it is not benchmarked.",
                    stage=5,
                    inputs={"role_intent": "Profiling / Demographic"},
                ),
            )

        # Default — requires LLM to determine Candidate for Library
        return TagResult(
            value="Custom / One-off",
            source="llm",
            confidence=0.50,
            evidence=ev.fallback(
                "question.flow_reusability.deferred_to_llm",
                f"No rule fired: the scale "
                f"({question.scale_fingerprint or 'no fingerprint'}) matches no "
                f"benchmark, the options are not icon-based, and role_intent is "
                f"{role or 'unset'} rather than segmentation or demographic. Deciding "
                "whether the wording is generic enough for the question library needs "
                "the LLM — this 0.50 placeholder is for Call 2 to replace.",
                stage=5,
                inputs={"scale_fingerprint":
                            question.scale_fingerprint or "(none)",
                        "role_intent": role or "(unset)",
                        "question_type": question.question_type},
            ),
        )


def create_tagger() -> FlowReusabilityTagger:
    return FlowReusabilityTagger()
