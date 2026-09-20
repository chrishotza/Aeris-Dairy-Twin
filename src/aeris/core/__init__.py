"""Core AERIS state logic."""

from .causes import PROBABLE_CAUSES, probable_causes, rank_probable_causes
from .interventions import (
    InterventionPlan,
    group_intervention_plan,
    intervention_plan,
)
from .mapping import SIGNAL_STATE_MAP, signal_family
from .state_engine import (
    WelfareSignals,
    classify_regime,
    group_burden,
    severity_score,
    state_from_signals,
    structural_validity,
    unit_burden,
)
from .transitions import (
    TransitionEvidence,
    transition_is_allowed,
    transition_label,
)

__all__ = [
    "WelfareSignals",
    "classify_regime",
    "group_burden",
    "severity_score",
    "state_from_signals",
    "structural_validity",
    "unit_burden",
    "TransitionEvidence",
    "transition_is_allowed",
    "transition_label",
    "PROBABLE_CAUSES",
    "probable_causes",
    "rank_probable_causes",
    "InterventionPlan",
    "group_intervention_plan",
    "intervention_plan",
    "SIGNAL_STATE_MAP",
    "signal_family",
]
