# AERIS architecture reconstructed from the corpus

## 1. Problem framing

The corpus describes AERIS as a digital-twin-oriented welfare intelligence system intended to replace fragmented and periodic monitoring with continuous, multimodal state estimation and operational escalation.

## 2. Three simultaneous levels

The documented system operates at:

- **animal level**
- **pen / group level**
- **production-unit level**

At the group level, animal concern states are aggregated to identify local clusters, shared stressors and deterioration patterns.

At the unit level, concern rates and escalation logic are summarized for producers and veterinarians.

## 3. Structural validity

The documented core introduces a structural-validity variable V in [0,1].

The stated purpose is to separate noise from meaningful deterioration.

The corpus describes validity as increasing when:

- directional deterioration is present;
- persistence through time is present;
- cross-signal coherence is present.

## 4. Severity

The documented severity variable S is derived from:

- activity drop;
- rumination drop;
- locomotion drop;
- heat rise.

## 5. Regime engine

The core regime set is:

- stable
- transition
- collapse

The operational visualization maps these states to:

- GREEN
- YELLOW
- RED

The corpus also uses language such as stable / transition / collapse-risk when describing interpretation.

## 6. Hierarchical burden

Documented equations:

B_group = n_transition + 2 * n_collapse

B_unit = groups_transition + 2 * groups_collapse

The purpose is to make deterioration aggregate upward from animal → group → unit.

## 7. Multimodal inputs

The broader AERIS architecture describes fusion of:

- activity;
- rumination-related indicators;
- locomotion quality;
- respiratory load;
- thermal discomfort;
- feed/water stress;
- ventilation evidence;
- management disruption;
- visual anomaly proxies.

## 8. Operational layers

The corpus contains separate concepts for:

- alert generation;
- intervention logic;
- probable-cause mapping;
- signal-to-state mapping;
- state transitions;
- structural validity;
- multiregime classification;
- dashboard outputs.

This repository initially implements only the documented mathematical core. Other layers are being reconstructed in later commits.

## 9. Important scope boundary

The corpus explicitly distinguishes a simulation-backed integrated proof of concept from a fully field-validated commercial product. This repository preserves that distinction.
