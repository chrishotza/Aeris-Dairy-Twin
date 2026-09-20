# AERIS release readiness

## Current status

AERIS is now at a **research release-candidate** stage.

The repository has a clean public implementation, documented scope and non-claims, explicit operational logic, a benchmark bridge, preserved historical research scripts, and a passing GitHub Actions test run.

This status is different from field validation or production readiness.

## Green — already in place

- Clean `src/aeris/` research API for state estimation, hierarchy, alerts, simulation and benchmark adapters.
- Explicit GREEN/YELLOW/RED transitions.
- Explicit probable-cause hypotheses and intervention plans.
- CVB projection and channel diagnostics.
- 66 historical research scripts preserved under `experiments/historical/`.
- Synthetic validation metrics documented with provenance.
- Repository boundary separating original code from third-party raw data.
- Public scope and non-claims documentation.
- GitHub Actions test workflow passing on the current main commit.

## Yellow — final engineering work before a polished public research release

### 1. Deterministic public demonstration

Add a small, license-safe synthetic fixture and checked-in example outputs covering:

- animal trajectory;
- group trajectory;
- unit trajectory;
- alert feed;
- recommended action;
- one reproducibility manifest.

The fixture should be generated from scratch and clearly labeled synthetic.

### 2. Champion reproducibility

Convert the historical ventilation-refinement/champion search into a clean configurable experiment that accepts explicit input paths and writes:

- configuration;
- seed;
- parameter leaderboard;
- best parameters;
- metrics;
- scenario breakdown;
- output checksums.

The historical scripts remain provenance; the clean experiment should be the reproducible public entry point.

### 3. Reproducibility report

Add one report that ties:

`configuration -> source commit -> input identity -> seed -> run -> metrics -> artifacts`

together, so a reader can reconstruct exactly what produced a reported result.

### 4. Evidence packaging

Make the public repository self-contained enough that a reviewer does not have to reconstruct the intended dashboard, alert examples, state trajectory and simulated data format from prose alone.

## Red / external gate

### Challenge-derived public release boundary

The corpus contains an InoCrowd Ownership of R&D Results agreement. Its text states a 90-day Exclusivity Period from the challenge deadline, an exclusive option during that period, restrictions on granting/disclosing/transferring the proposed solution or work product during the exclusivity period, and continuing confidentiality obligations.

Before making a new external release, submission, patent filing, or publication that incorporates challenge-specific work product, verify the applicable challenge deadline, whether the exclusivity period has ended, whether an option was exercised, and which material is actually covered.

This repository already keeps the agreement itself and third-party raw data out of the public tree.

## Scientific evidence boundary

The integrated AERIS proof remains simulation-backed. The repository must not present the pilot placeholder metrics as executed field results.

External integrated field validation, live streaming, production alert routing and broad farm deployment remain future work.

## Release gate

A polished public research release can be considered ready when the Yellow items are closed and the Challenge-derived public-release boundary has been checked.

Field validation is a separate evidence milestone; it is not silently treated as complete by releasing the research code.
