# Experiments reconstructed from the corpus

The original AERIS project contains multiple generations of simulation and validation scripts, including:

- parameter sweeps;
- channel sweeps;
- balanced sweeps;
- upstream sweeps;
- reinforced simulations;
- ventilation refinement;
- alert-quality validation;
- lead-time validation;
- group emergence validation;
- unit emergence validation;
- structural-validity validation;
- synthetic full-pipeline runs.

The documented integrated champion is challenge_ventilation_refinement_v3.

## Reproducibility plan

The public repository will progressively replace the original file-dump organization with explicit experiment entry points.

Each experiment should eventually record:

- configuration;
- inputs;
- random seed where applicable;
- generated artifacts;
- metrics;
- exact source commit;
- whether data are synthetic, benchmark-derived or field-derived.

The first public reconstruction intentionally avoids importing the entire historical artifact tree unchanged.
