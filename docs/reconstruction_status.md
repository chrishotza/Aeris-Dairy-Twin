# AERIS reconstruction status

This repository is being rebuilt from the 2026 AERIS corpus dump.

## Corpus preservation

The source corpus contains 66 Python research scripts under the 10_code research directory.

All 66 historical Python scripts are now preserved under experiments/historical/.

This includes simulation, sweeps, channel diagnostics, upstream/threshold searches, CVB extraction and projection, alert-feed construction/calibration, incident management, validation, comparison, evidence-pack and submission-pack builders.

Historical scripts are retained as provenance artifacts. They are not automatically treated as production-ready modules.

## Reconstructed

- package skeleton;
- documented core state equations;
- animal/group/unit burden logic;
- clean multimodal simulation API;
- alert-generation API;
- compact synthetic validation logic;
- documented validation metrics;
- integrated champion metrics;
- challenge-response map;
- data/provenance policy;
- scope and non-claims;
- benchmark bridge for CVB behavior tables;
- channel diagnostics;
- portable data manifests;
- historical experimental scripts with provenance.

## Current clean pipeline

The public implementation now exposes:

1. aeris.simulation.simulate(...)
2. aeris.aggregation.aggregate_group(...)
3. aeris.aggregation.aggregate_unit(...)
4. aeris.alerts.build_alert_feed(...)
5. aeris.simulation.evaluate.evaluate_units(...)
6. aeris.benchmark.project_cvb_behavior_table(...)
7. aeris.benchmark.diagnose_channels(...)
8. aeris-sim
9. aeris-benchmark

The clean implementation removes dependence on the original local filesystem layout.

## Historical vs reconstructed

Historical scripts remain under experiments/historical/.

The clean src/aeris/ implementation is the reproducible research API intended for future extension. Historical scripts preserve the original experimental procedures and filesystem-oriented workflows.

## Remaining high-value reconstruction targets

1. Reconstruct the champion search/refinement loop as a configurable experiment.
2. Reconstruct the CVB alert-feed/calibration chain as a clean benchmark module.
3. Reconstruct full-pipeline and incident-management orchestration.
4. Expose state transitions, probable-cause mapping and intervention logic as explicit APIs.
5. Add deterministic result fixtures and result checksums.
6. Build a small, license-safe demonstration dataset for the public repository.
7. Add a reproducibility report that ties configuration -> run -> metrics -> artifacts together.

## Evidence boundary

The corpus contains synthetic validation, benchmark-derived intermediate evidence and pilot placeholders. The repository must not present pilot placeholders as executed field results. Real-world field validation, live streaming, production alert routing and wider deployment remain open work.
