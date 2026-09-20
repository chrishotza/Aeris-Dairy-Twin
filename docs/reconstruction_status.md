# AERIS reconstruction status

This repository is being rebuilt from the 2026 AERIS corpus dump.

## Reconstructed

- package skeleton;
- documented core state equations;
- animal/group/unit burden logic;
- synthetic validation logic;
- documented validation metrics;
- integrated champion metrics;
- challenge-response map;
- data/provenance policy;
- scope and non-claims.

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

## Important distinction

The historical scripts were developed against a particular local directory structure and a large corpus of generated and external artifacts. They should not be copied into the public package unchanged until their dependencies, provenance and data assumptions have been made explicit.

The public reconstruction therefore has two layers:

1. AERIS Core — clean, inspectable research implementation.
2. Historical Experiments — progressively ported experiment logic with provenance and reproducibility notes.

## Current next reconstruction targets

- port the integrated simulator into src/aeris/simulation/;
- port alert generation into src/aeris/alerts/;
- port group/unit aggregation into src/aeris/aggregation/;
- create small deterministic synthetic fixtures;
- add experiment configuration files;
- reproduce the documented champion metrics without third-party raw datasets.
