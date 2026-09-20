# AERIS reconstruction status

This repository was rebuilt from the 2026 AERIS corpus dump.

## Corpus preservation

The source corpus contains 66 Python research scripts under the 10_code research directory.

All 66 historical Python scripts are preserved under
\`experiments/historical/\`.

Historical scripts are provenance artifacts first. They depend on the original
research directory structure and generated datasets, so they are not
automatically treated as production-ready modules.

## Reconstructed public implementation

The clean public implementation exposes:

1. \`aeris.simulation.simulate(...)\`
2. \`aeris.aggregation.aggregate_group(...)\`
3. \`aeris.aggregation.aggregate_unit(...)\`
4. \`aeris.alerts.build_alert_feed(...)\`
5. \`aeris.simulation.evaluate.evaluate_units(...)\`
6. \`aeris.benchmark.project_cvb_behavior_table(...)\`
7. \`aeris.benchmark.diagnose_channels(...)\`
8. \`aeris-sim\`
9. \`aeris-benchmark\`

It also exposes explicit state transitions, probable-cause mapping, signal
mapping and intervention logic.

## Release-candidate additions

The technical reproducibility layer now includes:

- \`experiments/reproducibility_demo.py\`;
- \`data/synthetic/demo/\`;
- \`experiments/champion_refinement.py\`;
- \`docs/reproducibility.md\`;
- deterministic tests for the public fixture and champion-search behavior.

The clean champion experiment is a public reconstruction of the historical
search structure. It is intentionally smaller by default than the historical
800 + 1600 evaluation schedule.

## Remaining boundaries

The remaining open work is outside the basic repository engineering layer:

- challenge/IP disclosure verification;
- executed external field validation;
- live/production deployment evidence.

These are not silently treated as complete by the presence of the release
artifacts.
