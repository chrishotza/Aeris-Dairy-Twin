# AERIS experiment catalog extracted from the corpus

The dump identifies the following historical experiment families.

## Integrated simulation

- run_aeris_challenge_multimodal_sim_v1.py
- run_full_pipeline_v1.py
- run_full_pipeline_synth_v2.py

The integrated simulator is documented as combining behavioural, biomechanical, physiological, environmental and management channels and producing animal/group/unit trajectories and alert-oriented outputs.

## Champion search and refinement

- run_challenge_sim_reinforced_v2.py
- run_challenge_balanced_sweep_v2.py
- run_challenge_channel_sweep_v1.py
- run_challenge_channel_sweep_v2_constrained.py
- run_challenge_threshold_sweep_v1.py
- run_challenge_upstream_sweep_v1.py
- run_challenge_ventilation_refinement_v3.py

The ventilation refinement script uses randomized parameter sampling followed by refinement around top seeds, with explicit viability gates and a multi-metric objective.

The historical V3 script records:

- random seed 2097;
- 800 random evaluations;
- 80 top seeds;
- 1600 refinement evaluations;
- persistence parameters for RED and YELLOW;
- scenario-specific recall tracking;
- baseline concern/specificity protection.

## CVB / real-data bridge

The corpus includes:

- run_cvb_param_sweep_v1.py
- run_cvb_param_sweep_v2.py
- real_data_loader_v1.py
- project_cvb_to_aeris_v1.py
- project_cvb_to_aeris_v2.py
- CVB inspection and mapping scripts;
- visibility-gated alert feed generation;
- manual adjudication/review pack generation.

The real-data target notes identify MmCows as the primary target and CowScreeningDB as a secondary narrower validation path.

## Validation stack

- validate_structural_validity_v1.py
- validate_group_emergence_v1.py
- validate_unit_emergence_v1.py
- validate_unit_emergence_v2.py
- validate_leadtime_v1.py
- validate_leadtime_v2.py
- validate_alert_quality_v1.py

The corpus reports a progression from animal-level state separation through group/unit emergence and finally early-warning/alert-quality evaluation.
