# AERIS validation evidence from the source corpus

This document records validation results that are explicitly present in the AERIS corpus.

## Synthetic validation core

The corpus reports:

### Validation 1 — animal-level structural validity + regime

- accuracy = 0.927
- stable separated perfectly
- collapse separated perfectly
- transition partially overlaps with collapse

### Validation 2 — group emergence

- accuracy = 0.908
- stable separated perfectly
- collapse separated perfectly
- transition remains the hardest boundary

### Validation 3 — unit emergence

The initial aggregation rule failed on transition. The documented correction used burden-based aggregation.

- final accuracy = 0.887

### Validation 4 — early warning / lead time

The corpus documents an initial sensitivity problem with yellow alerts and a correction using a minimum transition severity threshold.

Reported values:

- false yellow rate on stable = 0.017
- detection rate yellow = 1.000
- mean lead time yellow = 6.15
- red = late high-risk confirmation

### Validation 5 — alert quality

Reported values:

- precision = 0.970
- recall = 1.000
- specificity = 0.937
- F1 = 0.985

## Integrated simulation-backed champion

The current integrated champion is documented as:

`challenge_ventilation_refinement_v3`

Reported metrics:

- precision = 0.6998
- recall = 0.8862
- specificity = 0.7834
- accuracy = 0.8207
- stable-baseline concern rate = 0.0107
- stable-baseline specificity = 0.9893

Scenario-family recall:

- ventilation = 0.9779
- heat stress = 0.9797
- lameness / locomotion = 0.9749
- feed disruption = 0.9350
- water stress = 0.9543

## Validation boundary

The corpus explicitly states that the integrated proof remains simulation-backed and that external integrated field validation is a future pilot objective.

This repository therefore does not represent the integrated system as field validated.
