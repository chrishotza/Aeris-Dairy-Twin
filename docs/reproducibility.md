# AERIS reproducibility guide

## Purpose

The public repository now includes a deterministic synthetic fixture and a
clean, configurable reconstruction of the champion-search structure.

This closes the main reproducibility packaging gap without redistributing
third-party videos, annotations, challenge archives, or field data.

## 1. Verify the synthetic fixture

Install the package:

\`\`\`bash
python -m pip install -e ".[test]"
python -m pytest -q
\`\`\`

Regenerate the public fixture:

\`\`\`bash
python experiments/reproducibility_demo.py --output data/synthetic/demo
\`\`\`

Compare the regenerated CSV SHA256 values with:

\`\`\`text
data/synthetic/demo/manifest.json
\`\`\`

The fixture is deterministic and synthetic. Its expected Unit_1 path is:

\`\`\`text
GREEN -> GREEN -> YELLOW -> RED -> YELLOW -> GREEN
\`\`\`

## 2. Reproduce a clean simulation input

The public simulator can generate an animal-state table without any external
dataset:

\`\`\`bash
python -m aeris.cli \
  --hours 48 \
  --units 2 \
  --groups 2 \
  --animals 4 \
  --seed 2042 \
  --output results/repro_sim
\`\`\`

The resulting:

\`\`\`text
results/repro_sim/animal_states.csv
\`\`\`

is compatible with the public champion-refinement experiment.

## 3. Run champion refinement

Run the compact public search:

\`\`\`bash
python experiments/champion_refinement.py \
  --input results/repro_sim/animal_states.csv \
  --output results/champion_refinement \
  --seed 2097 \
  --random-iters 80 \
  --top-seeds 12 \
  --refine-iters 160
\`\`\`

Outputs:

- \`leaderboard.csv\`
- \`scenario_metrics.csv\`
- \`best_params.json\`
- \`manifest.json\`

The search follows the structural pattern documented by the historical
reinforced and ventilation-refinement scripts: randomized parameter search,
top-seed selection, local refinement, scenario metrics, persistence, and
viability constraints.

## 4. Historical-reproduction boundary

The public experiment is explicitly a reconstruction. It is not a claim of
byte-for-byte reproduction of the historical challenge filesystem.

The corpus documents the historical V3 refinement as using seed 2097,
800 random evaluations, 80 top seeds and 1600 refinement evaluations.
The public experiment uses smaller defaults so that a reviewer can execute
it locally and in CI; the CLI allows larger counts when the required input
artifacts are available.

The historical champion metrics recorded elsewhere in the repository remain
source-corpus evidence. They are not silently re-labeled as the output of
this compact public reconstruction.

## 5. Evidence boundary

The public fixture and simulation are research/synthetic artifacts.
Integrated field validation, live streaming, production routing and broad
farm deployment are separate milestones.

Third-party raw data must remain outside the repository unless redistribution
rights are independently established.
