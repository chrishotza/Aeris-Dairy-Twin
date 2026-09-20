"""Reconstructed AERIS core state logic.

This module implements the mathematical structure documented in the
AERIS research corpus. It is intentionally small and transparent so the
research logic can be inspected and extended.

Documented source concepts:
- structural validity V in [0, 1]
- severity S
- regime R in {stable, transition, collapse}
- group burden B_group = n_transition + 2*n_collapse
- unit burden B_unit = groups_transition + 2*groups_collapse

The repository does not claim that these reconstructed functions are
identical to every historical experimental script in the original corpus.
They are an explicit open implementation of the documented core.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WelfareSignals:
    """Single animal signal vector.

    Values are normalized/scaled inputs supplied by the caller.
    The documented core uses activity, rumination, locomotion and heat.
    """

    activity: float
    rumination: float
    locomotion: float
    heat: float


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def _validate_finite(values: Sequence[float]) -> None:
    if any(not isfinite(float(v)) for v in values):
        raise ValueError("all signal values must be finite")


def structural_validity(
    directional_deterioration: float,
    persistence: float,
    cross_signal_coherence: float,
) -> float:
    """Compute documented structural-validity score V in [0, 1].

    The corpus describes V as increasing with:
    - directional deterioration,
    - persistence through time,
    - cross-signal coherence.

    This open implementation uses their arithmetic mean.
    """

    values = (
        float(directional_deterioration),
        float(persistence),
        float(cross_signal_coherence),
    )
    _validate_finite(values)

    score = _mean(values)
    return max(0.0, min(1.0, score))


def severity_score(
    activity_drop: float,
    rumination_drop: float,
    locomotion_drop: float,
    heat_rise: float,
) -> float:
    """Compute the documented severity score S.

    The corpus defines severity from:
    - activity drop
    - rumination drop
    - locomotion drop
    - heat rise

    This open implementation uses their arithmetic mean.
    """

    values = (
        float(activity_drop),
        float(rumination_drop),
        float(locomotion_drop),
        float(heat_rise),
    )
    _validate_finite(values)

    return _mean(values)


def classify_regime(
    validity: float,
    severity: float,
    *,
    transition_validity_threshold: float = 0.5,
    collapse_validity_threshold: float = 0.5,
    transition_severity_threshold: float = 0.35,
    collapse_severity_threshold: float = 0.70,
) -> str:
    """Map validity/severity to stable, transition or collapse.

    Thresholds are explicit parameters because the corpus describes them as
    calibration-dependent. These defaults are a transparent research
    implementation, not a claim that they are the final production values.
    """

    validity = float(validity)
    severity = float(severity)

    if not 0.0 <= validity <= 1.0:
        raise ValueError("validity must be in [0, 1]")
    if not isfinite(severity):
        raise ValueError("severity must be finite")

    if (
        validity >= collapse_validity_threshold
        and severity >= collapse_severity_threshold
    ):
        return "collapse"

    if (
        validity >= transition_validity_threshold
        and severity >= transition_severity_threshold
    ):
        return "transition"

    return "stable"


def group_burden(
    n_transition: int,
    n_collapse: int,
) -> int:
    """Compute documented group burden.

    B_group = n_transition + 2 * n_collapse
    """

    if n_transition < 0 or n_collapse < 0:
        raise ValueError("counts must be non-negative")

    return int(n_transition) + 2 * int(n_collapse)


def unit_burden(
    groups_transition: int,
    groups_collapse: int,
) -> int:
    """Compute documented unit burden.

    B_unit = groups_transition + 2 * groups_collapse
    """

    if groups_transition < 0 or groups_collapse < 0:
        raise ValueError("counts must be non-negative")

    return int(groups_transition) + 2 * int(groups_collapse)


def state_from_signals(
    *,
    signals: WelfareSignals,
    validity: float,
    activity_drop: float,
    rumination_drop: float,
    locomotion_drop: float,
    heat_rise: float,
    **thresholds: float,
) -> dict[str, float | str]:
    """Convenience function returning a documented operational state."""

    values = [
        signals.activity,
        signals.rumination,
        signals.locomotion,
        signals.heat,
    ]
    _validate_finite(values)

    severity = severity_score(
        activity_drop=activity_drop,
        rumination_drop=rumination_drop,
        locomotion_drop=locomotion_drop,
        heat_rise=heat_rise,
    )
    regime = classify_regime(validity, severity, **thresholds)

    color = {
        "stable": "GREEN",
        "transition": "YELLOW",
        "collapse": "RED",
    }[regime]

    return {
        "validity": float(validity),
        "severity": float(severity),
        "regime": regime,
        "color": color,
    }
