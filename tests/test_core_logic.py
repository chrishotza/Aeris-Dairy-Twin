import pytest

from aeris.core.causes import probable_causes, rank_probable_causes
from aeris.core.interventions import intervention_plan
from aeris.core.mapping import signal_family
from aeris.core.transitions import TransitionEvidence, transition_is_allowed


def test_green_to_yellow_transition():
    evidence = TransitionEvidence(
        early_anomaly_rise=True,
        directional_deterioration=True,
        sufficient_persistence=True,
        valid_structural_state=True,
    )
    assert transition_is_allowed(
        "GREEN", "stable", "YELLOW", "transition", evidence
    )


def test_red_requires_validity_and_recovery_is_explicit():
    evidence = TransitionEvidence(
        anomaly_escalation=True,
        cross_signal_agreement=True,
        sufficient_persistence=True,
        action_threshold_exceeded=True,
        valid_structural_state=False,
    )
    assert not transition_is_allowed(
        "YELLOW", "transition", "RED", "collapse-risk", evidence
    )

    recovery = TransitionEvidence(
        post_intervention_improvement=True,
        anomaly_decreasing=True,
        recovery_sustained=True,
        valid_structural_state=True,
    )
    assert transition_is_allowed(
        "RED",
        "collapse-risk",
        "YELLOW",
        "recovery-transition",
        recovery,
    )


def test_probable_causes_are_hypotheses():
    causes = probable_causes("heat stress signal")
    assert "poor ventilation" in causes
    ranked = rank_probable_causes(["heat_stress_signal", "low_activity"])
    assert ranked


def test_intervention_and_mapping_are_source_defined():
    plan = intervention_plan("RED", "collapse-risk")
    assert plan.priority == "high / critical"
    mapping = signal_family("thermal environmental")
    assert "THI rise" in mapping["inputs"]


def test_unknown_state_is_rejected():
    with pytest.raises(KeyError):
        intervention_plan("BLUE", "unknown")
