"""Cadence tagger: statistical temporal analysis of response patterns."""

from __future__ import annotations

import statistics
from datetime import datetime

from models import evidence as ev
from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

_MIN_RESPONSES_FOR_ANALYSIS = 5


class CadenceTagger(ProjectTagger):
    name = "project.cadence"
    tag_dimension = "survey_cadence"
    stage = 2
    source_type = "statistical"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        stats = context.response_stats

        if stats is None or stats.total_responses < _MIN_RESPONSES_FOR_ANALYSIS:
            observed = stats.total_responses if stats else 0
            return TagResult(
                value="Ad-hoc",
                source="statistical",
                confidence=0.60,
                evidence=ev.statistic(
                    "project.cadence.insufficient_responses",
                    f"Only {observed} response(s) — below the {_MIN_RESPONSES_FOR_ANALYSIS} "
                    "needed to read a temporal pattern at all. Ad-hoc here is the "
                    "can't-tell answer, not a measured cadence.",
                    measure="total_responses",
                    observed=observed,
                    threshold=_MIN_RESPONSES_FOR_ANALYSIS,
                    stage=2,
                ),
            )

        timestamps = stats.response_timestamps
        span = stats.span_days

        if span <= 14:
            return TagResult(
                value="One-time",
                source="statistical",
                confidence=0.90,
                evidence=ev.statistic(
                    "project.cadence.short_span",
                    f"All {stats.total_responses} responses landed inside a {span}-day "
                    "window (≤14), which is a single fielding rather than a repeating "
                    "programme.",
                    measure="span_days",
                    observed=span,
                    threshold=14,
                    stage=2,
                    inputs={"total_responses": stats.total_responses},
                ),
            )

        # Compute inter-response gaps
        gaps_days = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 86400
            for i in range(len(timestamps) - 1)
        ]

        if not gaps_days:
            return TagResult(
                value="Ad-hoc", source="statistical", confidence=0.60,
                evidence=ev.statistic(
                    "project.cadence.no_gaps_computable",
                    "The response set spans more than 14 days but yields no "
                    "inter-response gaps to measure, so no cadence can be read.",
                    measure="inter_response_gaps",
                    observed=0,
                    stage=2,
                    inputs={"span_days": span,
                            "total_responses": stats.total_responses},
                ),
            )

        median_gap = statistics.median(gaps_days)
        max_gap = max(gaps_days)

        # Always-on: continuous flow with small gaps
        if span > 90 and median_gap < 7 and max_gap < 30:
            return TagResult(
                value="Always-on",
                source="statistical",
                confidence=0.85,
                evidence=ev.statistic(
                    "project.cadence.continuous_flow",
                    f"Responses arrive continuously over a long window: {span} days of "
                    f"collection, a median gap of {median_gap:.1f} days between "
                    f"responses, and never a quiet stretch longer than {max_gap:.1f} "
                    "days. That is an always-on collector, not a fielded wave.",
                    measure="median_gap_days",
                    observed=round(median_gap, 1),
                    threshold=7,
                    stage=2,
                    inputs={"span_days": span,
                            "max_gap_days": round(max_gap, 1),
                            "total_responses": stats.total_responses},
                ),
            )

        # Check invitation channel signal
        if context.invitation_signals:
            channels = context.invitation_signals.channel_distribution
            if "Shareable URL" in channels and span > 30:
                shareable_pct = channels["Shareable URL"] / max(context.invitation_signals.total_invitations, 1)
                if shareable_pct > 0.5:
                    return TagResult(
                        value="Always-on",
                        source="statistical",
                        confidence=0.75,
                        evidence=ev.statistic(
                            "project.cadence.shareable_url_dominant",
                            f"{shareable_pct:.0%} of invitations go out as a shareable "
                            f"URL rather than a targeted send, over a {span}-day span. "
                            "An open link that anyone can answer at any time behaves as "
                            "an always-on collector even when the response timing looks "
                            "irregular.",
                            measure="shareable_url_share",
                            observed=round(shareable_pct, 2),
                            threshold=0.5,
                            stage=2,
                            inputs={"span_days": span,
                                    "total_invitations":
                                        context.invitation_signals.total_invitations},
                        ),
                    )

        # Check for periodic clusters (recurring pattern)
        clusters = _detect_clusters(timestamps)
        if len(clusters) >= 2:
            cluster_gaps = [
                (clusters[i + 1][0] - clusters[i][-1]).days
                for i in range(len(clusters) - 1)
            ]
            if cluster_gaps:
                gap_variance = (
                    statistics.stdev(cluster_gaps) / statistics.mean(cluster_gaps)
                    if len(cluster_gaps) > 1 and statistics.mean(cluster_gaps) > 0
                    else 0
                )
                if gap_variance < 0.5:  # Fairly regular spacing
                    mean_gap = statistics.mean(cluster_gaps)
                    return TagResult(
                        value="Recurring",
                        source="statistical",
                        confidence=0.80,
                        evidence=ev.statistic(
                            "project.cadence.regular_clusters",
                            f"Responses fall into {len(clusters)} distinct bursts spaced "
                            f"about {mean_gap:.0f} days apart, and that spacing is "
                            f"regular (coefficient of variation {gap_variance:.2f}, "
                            "below the 0.5 cut-off). Repeated evenly-spaced waves is "
                            "what a recurring programme looks like.",
                            measure="cluster_gap_variation",
                            observed=round(gap_variance, 2),
                            threshold=0.5,
                            stage=2,
                            inputs={"cluster_count": len(clusters),
                                    "mean_cluster_gap_days": round(mean_gap),
                                    "span_days": span},
                        ),
                    )

        return TagResult(
            value="Ad-hoc",
            source="statistical",
            confidence=0.65,
            evidence=ev.statistic(
                "project.cadence.no_pattern",
                f"There are enough responses to analyse over a {span}-day span, but "
                f"they are neither continuous enough to be always-on (median gap "
                f"{median_gap:.1f}d, longest quiet stretch {max_gap:.1f}d) nor "
                "regularly clustered enough to be recurring. Ad-hoc is the measured "
                "answer here, not a fallback.",
                measure="span_days",
                observed=span,
                stage=2,
                inputs={"median_gap_days": round(median_gap, 1),
                        "max_gap_days": round(max_gap, 1),
                        "total_responses": stats.total_responses},
            ),
        )


def _detect_clusters(
    timestamps: list[datetime],
    gap_multiplier: float = 3.0,
) -> list[list[datetime]]:
    """Detect response clusters using gap-based segmentation.

    A gap > gap_multiplier × median_gap starts a new cluster.
    """
    if len(timestamps) < 3:
        return [timestamps]

    gaps = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    median_gap = statistics.median(gaps)
    threshold = median_gap * gap_multiplier

    clusters: list[list[datetime]] = [[timestamps[0]]]
    for i, gap in enumerate(gaps):
        if gap > threshold and threshold > 0:
            clusters.append([timestamps[i + 1]])
        else:
            clusters[-1].append(timestamps[i + 1])

    # Filter out tiny clusters (noise)
    return [c for c in clusters if len(c) >= 3]


def create_tagger() -> CadenceTagger:
    return CadenceTagger()
