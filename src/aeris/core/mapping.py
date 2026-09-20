"""Signal-family to state interpretation helpers."""

from __future__ import annotations


SIGNAL_STATE_MAP = {
    "locomotion": {
        "inputs": (
            "gait irregularity",
            "reduced mobility",
            "posture asymmetry",
        ),
        "interpretation": (
            "early discomfort",
            "lameness risk",
            "pain-related deterioration",
        ),
    },
    "rumination_feeding": {
        "inputs": (
            "rumination drop",
            "feeding visit reduction",
            "water visit irregularity",
        ),
        "interpretation": (
            "stress",
            "illness onset",
            "access problem",
            "environmental disruption",
        ),
    },
    "activity": {
        "inputs": (
            "low activity",
            "abnormal restlessness",
            "irregular lying/standing transitions",
        ),
        "interpretation": (
            "fatigue",
            "discomfort",
            "transition state",
            "welfare instability",
        ),
    },
    "thermal_environmental": {
        "inputs": (
            "THI rise",
            "heat stress index",
            "ventilation insufficiency",
        ),
        "interpretation": (
            "group-level stress",
            "unit-level instability",
            "collapse-risk escalation if persistent",
        ),
    },
    "group": {
        "inputs": (
            "competition proxy",
            "crowding proxy",
            "simultaneous anomaly rise across animals",
        ),
        "interpretation": (
            "management issue",
            "feeder/water access issue",
            "density problem",
            "environmental bottleneck",
        ),
    },
}


def signal_family(name: str) -> dict[str, tuple[str, ...]]:
    """Return source-defined signal inputs and interpretations."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return SIGNAL_STATE_MAP[key]
    except KeyError as exc:
        raise KeyError(f"unknown AERIS signal family: {name!r}") from exc
