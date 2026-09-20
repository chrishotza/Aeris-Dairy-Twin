"""AERIS probable-cause mapping reconstructed from the source corpus."""

from __future__ import annotations


PROBABLE_CAUSES: dict[str, tuple[str, ...]] = {
    "locomotion_degradation": (
        "lameness",
        "pain or discomfort",
        "floor / surface issue",
        "fatigue after stress event",
    ),
    "rumination_drop": (
        "illness onset",
        "feeding disruption",
        "heat stress",
        "water access problem",
        "post-handling stress",
    ),
    "low_activity": (
        "discomfort",
        "illness",
        "fatigue",
        "heat stress",
        "emerging welfare deterioration",
    ),
    "high_restlessness": (
        "environmental stress",
        "crowding / competition",
        "discomfort",
        "handling-related stress",
    ),
    "heat_stress_signal": (
        "high THI",
        "poor ventilation",
        "insufficient cooling",
        "water system under pressure",
    ),
    "group_instability": (
        "feeder access problem",
        "water access problem",
        "crowding",
        "bedding issue",
        "environmental bottleneck",
    ),
    "unit_level_deterioration": (
        "climate control failure",
        "water system issue",
        "feeding system issue",
        "management routine disruption",
        "multiple simultaneous group stressors",
    ),
}


def probable_causes(signal: str) -> tuple[str, ...]:
    """Return source-defined probable causes for a signal family.

    These are operational hypotheses, not diagnoses.
    """
    key = signal.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return PROBABLE_CAUSES[key]
    except KeyError as exc:
        raise KeyError(f"unknown AERIS signal family: {signal!r}") from exc


def rank_probable_causes(
    signals: list[str] | tuple[str, ...],
) -> list[tuple[str, float]]:
    """Count overlapping source-defined hypotheses across signal families.

    This is an aggregation helper for interpretability; it does not claim
    diagnostic probability.
    """
    counts: dict[str, float] = {}
    for signal in signals:
        for cause in probable_causes(signal):
            counts[cause] = counts.get(cause, 0.0) + 1.0

    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
