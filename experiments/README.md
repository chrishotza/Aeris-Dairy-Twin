# Experiments reconstructed from the corpus

The AERIS research history contains multiple generations of simulation and
validation scripts, including parameter sweeps, channel sweeps, balanced
searches, upstream searches, reinforced simulations, ventilation refinement,
alert-quality validation, lead-time validation, group/unit emergence and
synthetic full-pipeline runs.

## Public experiment entry points

### Deterministic public demo

\`\`\`bash
python experiments/reproducibility_demo.py --output data/synthetic/demo
\`\`\`

This creates the small synthetic fixture used by the public reproducibility
tests.

### Public champion refinement

\`\`\`bash
python experiments/champion_refinement.py \
  --input results/repro_sim/animal_states.csv \
  --output results/champion_refinement
\`\`\`

This is a configurable reconstruction of the historical reinforced /
ventilation-refinement search structure.

## Historical provenance

The documented integrated champion is
\`challenge_ventilation_refinement_v3\`.

Historical scripts remain under \`experiments/historical/\` with their original
names. They preserve provenance and experimental intent, but they depend on
the original data layout and generated artifacts.

Each clean public experiment should record:

- configuration;
- source/input identity;
- random seed;
- generated artifacts;
- metrics;
- exact source commit where available.
