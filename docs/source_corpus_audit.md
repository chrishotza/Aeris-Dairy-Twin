# AERIS source corpus audit

## What the corpus actually contains

The dump reports a source corpus of approximately 10,794 files and 2.38 GB. The key research area contains 66 Python scripts under the 10_code directory.

The code families include:

- integrated multimodal simulation;
- alert pipelines at animal, group and unit levels;
- balanced, channel, threshold, upstream and ventilation refinement searches;
- CVB extraction, inspection, mapping and projection;
- real-data manifests and preparation;
- alert-feed generation and calibration;
- manual review/adjudication tooling;
- incident-management pipelines;
- structural-validity, emergence, lead-time and alert-quality validation;
- comparison and evidence/submission pack builders.

## Repository coverage

All 66 historical Python scripts are now preserved in experiments/historical/.

The public clean implementation currently reconstructs the central state engine, simulation, hierarchy, alerts, CVB bridge, channel diagnostics and command-line interfaces.

The remaining gap is therefore no longer loss of the historical source stack. It is conversion of the most important historical procedures into stable, testable public APIs.

## Technical source notes that matter

The source architecture describes an ingestion -> feature extraction -> structural validity -> multiregime classification -> digital-twin state -> alerts -> recommended actions -> dashboard pipeline.

The structural-validity layer is intended to filter noise using persistence, directional consistency, limited structural degradation and cross-signal coherence.

The multiregime layer distinguishes basal/stable, transition and collapse-risk states.

The source also defines state transitions, probable-cause hypotheses and intervention logic. Those concepts are present in the corpus and should become explicit clean APIs rather than remaining implicit in historical scripts.

The data model covers animal, pen/group and unit/farm entities and includes welfare, environmental, operational and action fields.

## Validation boundary

The corpus reports a synthetic validation core covering animal, group, unit emergence, lead time and alert quality. It also contains a pilot_metrics_placeholder_v1.csv. That file is explicitly a placeholder and must not be represented as executed pilot evidence.

## Real-data boundary

The corpus identifies MmCows as the primary target and CowScreeningDB as a narrower secondary path. Raw third-party media and annotations are not part of the public repository unless their redistribution rights are established.

## Important historical/legal boundary

The corpus also contains an InoCrowd ownership/confidentiality agreement associated with the challenge. The public repository does not copy that agreement or the partner/submission material. Before relying on an open-source release for challenge-derived work, the applicable challenge terms and any accepted-submission restrictions should be checked separately.

## Conclusion

The original repository was not missing just one or two scripts. The major omission was preservation of the broader research stack. That gap has now been closed at the historical-source level.

The next engineering step is to turn the remaining high-value experimental families into configurable, reproducible public modules rather than copying more raw data or submission artifacts.
