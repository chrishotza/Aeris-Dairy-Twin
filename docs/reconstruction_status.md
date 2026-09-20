# AERIS reconstruction status

This repository is being rebuilt from the 2026 AERIS corpus dump.

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
- historical experimental scripts with provenance.

## Historical code inventory present in the source corpus

The source inventory identifies a substantial Python research stack under the 10_code directory, including:

- multimodal challenge simulation;
- animal/group/unit alert pipelines;
- balanced, channel, upstream and threshold sweeps;
- ventilation refinement;
- CVB parameter sweeps;
- CVB inspection and mapping;
- alert-quality validation;
- structural-validity validation;
- lead-time validation;
- group and unit emergence validation;
- evidence-pack builders.

## Current clean pipeline

The public implementation now exposes:

1. aeris.simulation.simulate(...)
2. aeris.aggregation.aggregate_group(...)
3. aeris.aggregation.aggregate_unit(...)
4. aeris.alerts.build_alert_feed(...)
5. aeris.simulation.evaluate.evaluate_units(...)
6. aeris-sim command-line execution

The clean simulator preserves the documented animal → group → unit architecture and the historical multimodal signal families while removing dependence on the original local filesystem layout.

## Historical vs reconstructed

Historical scripts remain under experiments/historical/.

They are provenance artifacts and preserve the original experimental logic. The clean src/aeris/ implementation is the reproducible research API intended for future extension.

## Current next reconstruction targets

- reproduce the integrated simulator outputs with committed deterministic fixtures;
- port channel diagnostics into src/aeris/benchmark/;
- port the CVB projection layer into a clean benchmark adapter;
- add benchmark dataset manifests and provenance;
- reconstruct the champion sweep as a configurable experiment;
- add reproducibility reports and result checksums.
