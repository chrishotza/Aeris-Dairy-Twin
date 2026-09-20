# AERIS — Open Multimodal State Engine for Dairy Welfare Monitoring

AERIS is an open research implementation of a multimodal dairy-welfare state engine reconstructed from the AERIS/InnoCrowd_DairyTwin research corpus.

The system is designed to transform noisy welfare-related signals into interpretable states and escalation signals at three levels:

- animal
- pen / group
- production unit / farm

## Core architecture

AERIS is organized around five documented layers:

1. **Structural validity** — separates noise from structurally meaningful deterioration.
2. **Severity** — estimates the magnitude of deterioration from changes in activity, rumination, locomotion and heat.
3. **Regime classification** — maps the state to stable, transition or collapse-risk regimes.
4. **Group burden** — aggregates animal deterioration into group-level operational load.
5. **Unit burden** — aggregates group deterioration into production-unit operational load.

The intended operational presentation uses:

- GREEN / stable
- YELLOW / transition
- RED / collapse-risk

## Evidence currently represented in the corpus

The source corpus documents:

- synthetic validation across animal, group and unit levels;
- early-warning and alert-quality experiments;
- a multimodal simulation-backed integrated champion;
- benchmark-informed visual work using real-world cattle video/annotation data;
- challenge-response and submission evidence packaging;
- pilot and deployment plans.

The current integrated champion is documented as `challenge_ventilation_refinement_v3`.

Documented champion metrics include:

| Metric | Result |
|---|---:|
| Precision | 0.6998 |
| Recall | 0.8862 |
| Specificity | 0.7834 |
| Accuracy | 0.8207 |
| Stable-baseline concern rate | 0.0107 |
| Stable-baseline specificity | 0.9893 |

Scenario-family recall documented in the corpus:

| Scenario | Recall |
|---|---:|
| Ventilation | 0.9779 |
| Heat stress | 0.9797 |
| Lameness / locomotion | 0.9749 |
| Feed disruption | 0.9350 |
| Water stress | 0.9543 |


## Quickstart

Install the package and test dependencies:

```bash
python -m pip install -e ".[test]"
pytest -q
```

Run the minimal state-estimation example:

```bash
python examples/quickstart.py
```

Run the compact synthetic validation:

```bash
python -m experiments.synthetic_validation
```

## Historical research code

Selected historical scripts from the source corpus are preserved under
`experiments/historical/` with their original logic and filenames.

These are provenance artifacts first. They depend on the original research
directory structure and generated datasets, so they are not represented as
drop-in production modules yet.

Important imported families include:

- integrated multimodal simulation;
- ventilation refinement V3;
- CVB parameter sweep;
- real-data loading and CVB-to-AERIS projection;
- animal, group and unit alert pipelines;
- structural-validity, group-emergence, lead-time and alert-quality validation.

See `docs/experiment_catalog.md` and `docs/reconstruction_status.md`.

## Core operational logic

Beyond the state score itself, AERIS now exposes source-defined operational logic for:

- explicit GREEN/YELLOW/RED state transitions;
- probable-cause hypotheses;
- signal-family interpretation;
- intervention plans for stable, transition, collapse-risk, recovery and invalid/hold states.

These APIs are reconstructed directly from the architecture notes in the source corpus. They are explicit research rules, not hidden business logic.

## Benchmark bridge

The clean benchmark layer now exposes:

- CVB behavior-table to AERIS projection;
- channel-head diagnostics against a stable baseline;
- portable raw-media manifests;
- benchmark I/O helpers;
- a command-line benchmark adapter.

Examples:

    aeris-benchmark cvb path/to/cvb_behavior_table.csv --output results/cvb
    aeris-benchmark channels path/to/animal_states.csv --output results/channels

The repository does not redistribute third-party raw datasets. Dataset names and acquisition targets are documented separately.

## Validation status

This repository distinguishes reconstructed/open research material from field deployment claims.

The integrated AERIS proof described in the corpus is **simulation-backed**. External integrated field validation, live streaming, production alert routing and broader farm deployment remain future validation work.

## Repository status

This repository is being reconstructed from the original AERIS corpus. Early commits intentionally prioritize:

- transparent architecture;
- reproducible core mathematics;
- documented validation results;
- synthetic/research-only examples;
- clean separation of third-party data from original project code.

## Data and third-party assets

The original corpus contains third-party datasets, large image collections, archives and generated artifacts. They are **not automatically redistributed here**.

See `docs/datasets.md` before adding any external dataset.

## Citation

See `CITATION.cff`.

## License

Original code in this repository is released under the 0BSD license. Third-party datasets and assets remain subject to their original terms.
