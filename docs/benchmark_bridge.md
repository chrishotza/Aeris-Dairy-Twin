# AERIS benchmark bridge

The source corpus identifies a real-data bridge based on CVB behavior annotations and identifies:

- MmCows as the primary real-data target;
- CowScreeningDB as a narrower secondary target focused on lameness and walking video.

The historical CVB-to-AERIS projection converts observation counts per clip and animal into behavioral proportions and four central proxies:

- activity;
- rest;
- anomaly;
- visibility.

It also computes a behavior-diversity count.

The documented projection states are:

- HOLD / low_visibility / collect_more_visual_evidence
- RED / behavior_anomaly / inspect_immediately
- YELLOW / behavior_shift / review_context_and_monitor
- GREEN / stable_observable / standard_monitoring

The clean implementation in src/aeris/benchmark/cvb.py preserves those source-defined rules.

## Channel diagnostics

The historical diagnostics script derives domain-specific channel heads for:

- heat;
- lameness;
- feed;
- water;
- ventilation;
- management;
- visual;
- base.

It compares each channel against a stable-baseline distribution using mean, standard deviation and upper quantiles, then ranks channels by separation.

The clean implementation in src/aeris/benchmark/channels.py preserves that analysis structure without tying it to the historical filesystem.

## Data provenance boundary

The source corpus explicitly names real datasets but does not by itself establish redistribution rights. This repository therefore contains adapters, manifests and preparation logic rather than third-party raw media.

Do not commit external raw video or image datasets unless their licensing permits redistribution.
