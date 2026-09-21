# AERIS technical release audit

**Audit date:** 2026-09-20  
**Repository:** `chrishotza/Aeris-Dairy-Twin`  
**Scope:** portfolio/CV-oriented technical research repository

## Engineering checks

- Clean public implementation under `src/aeris/`.
- Explicit animal/group/unit state logic, transitions, aggregation and alert generation.
- Benchmark bridge and CLI entry points present.
- All 66 historical research Python scripts preserved under `experiments/historical/`.
- Deterministic synthetic reproducibility fixture checked in under `data/synthetic/demo/`.
- Fixture manifest contains SHA256 checksums and the test suite verifies the checked-in hashes.
- Public champion-refinement reconstruction is parameterized, seeded and documented.
- Tests cover the public simulation, state engine, alerts, benchmark layer, CLI, reproducibility fixture and champion search.
- GitHub Actions test workflow is configured for pushes and pull requests.
- The latest completed release-work CI run before the checksum-test commit completed successfully.

## Evidence boundary

The repository distinguishes:

1. synthetic/reproducibility evidence;
2. natural-video benchmark evidence derived from the research corpus;
3. future external field validation.

Historical benchmark and champion metrics remain identified as source-corpus evidence. They are not presented as executed farm-pilot results.

Third-party raw datasets, challenge-room materials and the InoCrowd agreement are not redistributed in the public repository.

## Portfolio positioning

The repository is intended to demonstrate:

- research reconstruction from a large heterogeneous corpus;
- translation of exploratory research code into maintainable Python modules;
- explicit state-machine and alert logic;
- reproducible synthetic experimentation;
- benchmark/data-interface design;
- automated testing and GitHub Actions CI;
- evidence-aware technical documentation.

## Remaining non-engineering items

No additional repository engineering gate is required for the current CV/portfolio purpose.

Separate future work includes real-farm pilot validation and any contractual/IP review required before public disclosure of challenge-derived work product.
