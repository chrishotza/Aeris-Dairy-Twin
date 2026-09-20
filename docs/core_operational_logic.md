# AERIS core operational logic

The source corpus separates the state engine into distinct operational layers:

1. structural validity;
2. multiregime classification;
3. state transitions;
4. probable-cause hypotheses;
5. intervention logic.

This repository now exposes the last three as explicit, inspectable APIs.

## State transitions

The documented paths are:

- GREEN / STABLE -> YELLOW / TRANSITION;
- YELLOW / TRANSITION -> RED / COLLAPSE-RISK;
- RED / COLLAPSE-RISK -> YELLOW / RECOVERY-TRANSITION;
- YELLOW / RECOVERY-TRANSITION -> GREEN / STABLE;
- YELLOW / TRANSITION -> GREEN / STABLE;
- ANY STATE -> INVALID / HOLD.

The transition checks preserve the source requirements around persistence, structural validity, directional deterioration, cross-signal agreement, action thresholds, and sustained recovery.

## Probable causes

The source maps locomotion, rumination, activity, restlessness, heat, group instability and unit deterioration to possible operational causes.

These are explicitly treated as informed hypotheses, not final diagnoses.

## Intervention logic

The source defines operational actions for stable, transition, collapse-risk, recovery-transition and invalid/noisy states, plus escalations for multi-animal group deterioration and multi-group unit deterioration.

The implementation exposes those source-defined plans without adding unrecorded medical or diagnostic rules.

## Source boundary

These functions are reconstruction APIs derived from the project corpus. They are not claimed to be identical to every historical pipeline implementation.
