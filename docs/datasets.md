# Data policy and dataset map

The original AERIS corpus contains a mixture of original project artifacts, generated datasets and third-party research data.

## Reported corpus composition

The source inventory records approximately:

- 10,794 files
- 2,381.58 MB
- 9,000 JPG
- 531 JSON
- 487 Python files
- 182 CSV
- 20 PNG
- additional archives, PDFs, XML and other artifacts

## Large artifacts

The corpus inventory includes very large files, including:

- `2023-06-13_Sharma_Renuka_58916v1.zip` (~499.64 MB)
- `animal_states_challenge_sim_v1.csv` (~280.92 MB)
- `animal_states_reinforced_v2.csv` (~223.8 MB)
- `cvb_behavior_table_v1.csv` (~221.37 MB)
- `ava_train_set.csv` (~79.66 MB)

These are not copied into the repository during the initial reconstruction.

## Real-data targets documented in the corpus

Primary target:

- MmCows

Secondary target:

- CowScreeningDB

The source notes describe MmCows as a close fit because of dairy-cattle coverage and multimodal visual + wearable data. CowScreeningDB is described as a narrower bridge focused on lameness / walking video.

## Redistribution rule

Third-party data must only be added when its source terms explicitly permit redistribution.

For external datasets, this repository should store:

- acquisition instructions;
- source citation;
- expected directory layout;
- checksums where appropriate;
- preparation scripts.

Raw third-party datasets should remain outside the repository unless redistribution is clearly permitted.

## Synthetic data

Original synthetic data and small reproducibility fixtures can be added under `data/synthetic/`.

## Provenance

The source corpus includes SHA256 inventories and dataset metadata. Future reconstruction commits should preserve provenance where it can be traced unambiguously.
