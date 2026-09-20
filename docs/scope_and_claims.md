# AERIS scope, claims and non-claims

This document preserves the distinction made in the source corpus between what the system is intended to do and what it does not claim.

## Core claims documented in the corpus

1. AERIS shifts welfare monitoring from periodic/reactive assessment toward continuous, predictive and actionable intelligence.
2. The system operates simultaneously at animal, pen/group and unit/farm levels.
3. The documented technical novelty is not only multimodal fusion, but the combination of a structural-validity layer and a multiregime state engine.
4. The system converts noisy welfare-related signals into interpretable operational states such as GREEN/YELLOW/RED and stable/transition/collapse-risk.
5. The intended outcome is earlier intervention, clearer prioritization, traceability and scalable deployment.

## Explicit non-claims

The source corpus explicitly says AERIS:

- does not replace veterinarians, welfare experts or farm operators;
- does not claim perfect diagnosis;
- does not require a fully standardized sensor stack from day one;
- does not claim every anomaly should become an alert;
- does not claim immediate universal deployment;
- does not treat raw noise as evidence before structural validity;
- does not promise zero false alerts.

## Current evidence boundary

The integrated AERIS package is documented as a simulation-backed proof of concept with benchmark-informed visual evidence. External integrated field validation remains a future pilot objective.

## Research risk register retained from the corpus

### Data availability
Farms may expose different sensor and event data. The documented mitigation is support for partial and heterogeneous data stacks.

### False alerts
Too many weak alerts can reduce trust. The documented mitigation is structural-validity filtering before escalation.

### Interpretability
Operators need to understand why an alert was triggered. The documented mitigation is to pair alerts with probable cause, confidence and recommended action.

### Pilot access
A real dairy unit is required for integrated external validation. The corpus identifies partner outreach and a pilot checklist as mitigations.

### Domain validation
A technically coherent model may fail under real farm conditions. The documented strategy is pilot-first validation against observed welfare events.

### Complexity
Too many signals/layers may make deployment harder. The documented mitigation is to begin with a minimum viable stack.

### Scaling
Single-farm performance may not generalize. The documented mitigation is configurable adapters plus heterogeneous validation.

### Claim credibility
Overclaiming can weaken credibility. The documented mitigation is explicit separation between demonstrated evidence and future validation.
