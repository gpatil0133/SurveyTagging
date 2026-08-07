"""Cadence tagger: statistical temporal analysis of response patterns."""

from __future__ import annotations

import statistics
from datetime import datetime

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
            return TagResult(
                value="Ad-hoc",
                source="statistical",
                confidence=0.60,
                evidence=f"Insufficient responses ({stats.total_responses if stats else 0})",
            )

        timestamps = stats.response_timestamps
        span = stats.span_days

        if span <= 14:
            return TagResult(
                value="One-time",
                source="statistical",
                confidence=0.90,
                evidence=f"{stats.total_responses} responses in {span}-day span",
            )

        # Compute inter-response gaps
        gaps_days = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 86400
            for i in range(len(timestamps) - 1)
        ]

        if not gaps_days:
            return TagResult(value="Ad-hoc", source="statistical", confidence=0.60)

        median_gap = statistics.median(gaps_days)
        max_gap = max(gaps_days)

        # Always-on: continuous flow with small gaps
        if span > 90 and median_gap < 7 and max_gap < 30:
            return TagResult(
                value="Always-on",
                source="statistical",
                confidence=0.85,
                evidence=f"Span={span}d, median_gap={median_gap:.1f}d, max_gap={max_gap:.1f}d",
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
                        evidence=f"Shareable URL channel dominant ({shareable_pct:.0%})",
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
                    return TagResult(
                        value="Recurring",
                        source="statistical",
                        confidence=0.80,
                        evidence=f"{len(clusters)} clusters with ~{statistics.mean(cluster_gaps):.0f}d spacing",
                    )

        return TagResult(
            value="Ad-hoc",
            source="statistical",
            confidence=0.65,
            evidence=f"No clear temporal pattern (span={span}d)",
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
