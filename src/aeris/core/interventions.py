"""AERIS intervention logic reconstructed from the source corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionPlan:
    meaning: str
    actions: tuple[str, ...]
    priority: str


_PLANS = {
    ("GREEN", "stable"): InterventionPlan(
        meaning="normal operating condition",
        actions=("keep standard monitoring",),
        priority="low",
    ),
    ("YELLOW", "transition"): InterventionPlan(
        meaning="early warning / emerging instability",
        actions=(
            "inspect during routine check",
            "verify local context",
            "increase monitoring frequency",
        ),
        priority="medium",
    ),
    ("RED", "collapse-risk"): InterventionPlan(
        meaning="strong welfare deterioration / intervention required",
        actions=(
            "immediate targeted inspection",
            "prioritize veterinary or welfare response",
            "review environmental and management causes",
        ),
        priority="high / critical",
    ),
    ("YELLOW", "recovery-transition"): InterventionPlan(
        meaning="state improving after intervention",
        actions=(
            "continue monitoring",
            "confirm return to stable baseline",
        ),
        priority="medium until stable",
    ),
    ("INVALID", "hold"): InterventionPlan(
        meaning="insufficient structural validity",
        actions=(
            "do not escalate yet",
            "keep observing",
        ),
        priority="low",
    ),
}


def intervention_plan(zone: str, regime: str) -> InterventionPlan:
    """Return the source-defined intervention plan for an operational state."""
    key = (zone.upper(), regime.lower())
    try:
        return _PLANS[key]
    except KeyError as exc:
        raise KeyError(
            f"no source-defined intervention plan for {zone!r} / {regime!r}"
        ) from exc


def group_intervention_plan(yellow_animals: int, red_groups: int = 0) -> InterventionPlan:
    """Return group-level escalation guidance.

    The corpus defines a multi-animal YELLOW group condition as high priority.
    """
    if red_groups > 0:
        return InterventionPlan(
            meaning="likely unit/system-level instability",
            actions=(
                "activate unit-level response",
                "review climate, ventilation, water, feeding, operations",
            ),
            priority="critical",
        )

    if yellow_animals > 1:
        return InterventionPlan(
            meaning="likely group-level issue",
            actions=(
                "inspect feeder, water, density, heat, bedding",
            ),
            priority="high",
        )

    return intervention_plan("GREEN", "stable")
