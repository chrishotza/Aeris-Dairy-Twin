"""AERIS state-transition rules reconstructed from the source corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionEvidence:
    """Evidence used to evaluate a requested state transition."""

    early_anomaly_rise: bool = False
    directional_deterioration: bool = False
    sufficient_persistence: bool = False
    valid_structural_state: bool = False
    anomaly_escalation: bool = False
    cross_signal_agreement: bool = False
    action_threshold_exceeded: bool = False
    post_intervention_improvement: bool = False
    anomaly_decreasing: bool = False
    recovery_sustained: bool = False
    normalized_signals: bool = False
    stable_validity: bool = False
    no_active_escalation: bool = False
    baseline_restored: bool = False
    weak_anomaly_resolved: bool = False
    contradictory_signals: bool = False


def transition_is_allowed(
    current_zone: str,
    current_regime: str,
    target_zone: str,
    target_regime: str,
    evidence: TransitionEvidence,
) -> bool:
    """Evaluate the six source-defined transition paths.

    No RED transition is allowed without sufficient validity, and no
    recovery transition is allowed without sustained improvement.
    """
    current = (current_zone.upper(), current_regime.lower())
    target = (target_zone.upper(), target_regime.lower())

    if evidence.contradictory_signals and target[0] != "INVALID":
        return False

    if target == ("INVALID", "hold"):
        return True

    if current == ("GREEN", "stable") and target == ("YELLOW", "transition"):
        return (
            evidence.early_anomaly_rise
            and evidence.directional_deterioration
            and evidence.sufficient_persistence
            and evidence.valid_structural_state
        )

    if current == ("YELLOW", "transition") and target == ("RED", "collapse-risk"):
        return (
            evidence.anomaly_escalation
            and evidence.cross_signal_agreement
            and evidence.sufficient_persistence
            and evidence.action_threshold_exceeded
            and evidence.valid_structural_state
        )

    if current == ("RED", "collapse-risk") and target == (
        "YELLOW",
        "recovery-transition",
    ):
        return (
            evidence.post_intervention_improvement
            and evidence.anomaly_decreasing
            and evidence.recovery_sustained
            and evidence.valid_structural_state
        )

    if current == ("YELLOW", "recovery-transition") and target == ("GREEN", "stable"):
        return (
            evidence.normalized_signals
            and evidence.stable_validity
            and evidence.no_active_escalation
            and evidence.baseline_restored
        )

    if current == ("YELLOW", "transition") and target == ("GREEN", "stable"):
        return evidence.weak_anomaly_resolved and evidence.stable_validity

    return False


def transition_label(zone: str, regime: str) -> str:
    """Return the corpus naming convention for an operational state."""
    return f"{zone.upper()} / {regime.lower()}"
