# Synthetic data

This directory is reserved for small, reproducible synthetic fixtures derived from the documented AERIS state model.

The original corpus contains generated/synthetic CSV outputs for:

- animal states;
- group states;
- unit states;
- alert feeds;
- incident summaries;
- dashboard summaries.

Those large historical artifacts are intentionally not copied into the first public commit.

Future fixtures should be:

1. generated deterministically where possible;
2. small enough for source control;
3. clearly labeled synthetic;
4. accompanied by generation code;
5. traceable to the documented state definitions.
