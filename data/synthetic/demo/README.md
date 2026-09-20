# AERIS deterministic synthetic demo

This directory contains a tiny, license-safe fixture generated from scratch.

It is not field data, benchmark data, or a reproduction of a third-party
dataset.

## Contents

- \`animal_states.csv\` — 24 animal observations.
- \`group_states.csv\` — 12 group states.
- \`unit_states.csv\` — 12 unit states.
- \`alerts.csv\` — 8 escalation events.
- \`manifest.json\` — SHA256 checksums for the CSV fixture.
- \`reproducibility_report.json\` — expected state trajectory.

## Expected unit path

\`Unit_1\` follows:

\`GREEN -> GREEN -> YELLOW -> RED -> YELLOW -> GREEN\`

\`Unit_2\` remains GREEN.

## Regeneration

From the repository root:

\`\`\`bash
python experiments/reproducibility_demo.py --output data/synthetic/demo
\`\`\`

The regenerated CSV hashes should match \`manifest.json\`.
