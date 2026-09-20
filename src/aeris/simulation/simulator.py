"""Clean AERIS multimodal simulator reconstructed from the historical corpus.

The historical simulator combines behavioral, biomechanical, physiological,
environmental and management signals and propagates states through:

animal -> group -> unit.

This module extracts that architecture into an importable research API.
It deliberately keeps the historical scenario semantics while avoiding the
original filesystem/output coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

import numpy as np
import pandas as pd


SCENARIOS = (
    "stable_baseline",
    "heat_stress_wave",
    "lameness_cluster",
    "feeding_disruption",
    "water_system_issue",
    "ventilation_failure",
    "post_event_recovery",
)

SEVERITY_RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration corresponding to the documented simulator knobs."""

    monte_carlo: int = 1
    n_units: int = 4
    groups_per_unit: int = 4
    animals_per_group: int = 14
    hours: int = 96
    time_step_hours: int = 1
    measurement_noise: float = 0.035
    environment_noise: float = 0.04
    management_noise: float = 0.03
    yellow_animal_threshold: float = 0.42
    red_animal_threshold: float = 0.70
    yellow_group_share: float = 0.20
    red_group_share: float = 0.33
    yellow_unit_share: float = 0.15
    red_unit_share: float = 0.25
    seed: int = 2042


@dataclass(frozen=True)
class AnimalTraits:
    resilience: float
    heat_sensitivity: float
    lameness_sensitivity: float
    intake_sensitivity: float
    recovery_speed: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _scenario_profile(hour: int, hours: int) -> float:
    start = int(hours * 0.35)
    peak = int(hours * 0.55)
    end = int(hours * 0.78)

    if hour < start:
        return 0.0
    if start <= hour <= peak:
        return (hour - start) / max(1, peak - start)
    if peak < hour <= end:
        return 1.0 - 0.70 * ((hour - peak) / max(1, end - peak))
    return 0.0


def _scenario_modifiers(scenario: str, phase: float) -> dict[str, float]:
    out = {
        "thi_boost": 0.0,
        "ventilation_drop": 0.0,
        "water_drop": 0.0,
        "feed_drop": 0.0,
        "bedding_drop": 0.0,
    }

    if scenario == "heat_stress_wave":
        out["thi_boost"] = 0.40 * phase
        out["ventilation_drop"] = 0.10 * phase
    elif scenario == "feeding_disruption":
        out["feed_drop"] = 0.50 * phase
    elif scenario == "water_system_issue":
        out["water_drop"] = 0.55 * phase
    elif scenario == "ventilation_failure":
        out["ventilation_drop"] = 0.60 * phase
        out["thi_boost"] = 0.20 * phase
    elif scenario == "lameness_cluster":
        out["bedding_drop"] = 0.20 * phase
    elif scenario == "post_event_recovery":
        out["bedding_drop"] = 0.15 * phase
        out["feed_drop"] = 0.10 * phase

    return out


def _baseline_environment(
    rng: np.random.Generator,
    hour: int,
    noise: float,
) -> dict[str, float]:
    day_phase = math.sin((2 * math.pi * hour) / 24.0)
    temp = 23.0 + 6.0 * day_phase + rng.normal(0, 0.6 * noise / 0.035)
    humidity = 58.0 - 8.0 * day_phase + rng.normal(0, 1.0 * noise / 0.035)

    ventilation = _clamp01(0.82 + rng.normal(0, 0.04 * noise / 0.035))
    water = _clamp01(0.92 + rng.normal(0, 0.03 * noise / 0.035))
    feed = _clamp01(0.90 + rng.normal(0, 0.03 * noise / 0.035))
    bedding = _clamp01(0.86 + rng.normal(0, 0.03 * noise / 0.035))

    thi_raw = 0.7 * temp + 0.3 * (humidity / 2.0)
    thi_norm = _clamp01((thi_raw - 18.0) / 20.0)

    return {
        "temp_c": float(temp),
        "humidity_pct": float(humidity),
        "thi_norm": thi_norm,
        "ventilation_quality": ventilation,
        "water_status": water,
        "feed_delivery_quality": feed,
        "bedding_quality": bedding,
    }


def _animal_traits(rng: np.random.Generator) -> AnimalTraits:
    return AnimalTraits(
        resilience=_clamp01(rng.normal(0.62, 0.10)),
        heat_sensitivity=_clamp01(rng.normal(0.55, 0.12)),
        lameness_sensitivity=_clamp01(rng.normal(0.50, 0.15)),
        intake_sensitivity=_clamp01(rng.normal(0.52, 0.13)),
        recovery_speed=_clamp01(rng.normal(0.58, 0.10)),
    )


def _simulate_signals(
    *,
    rng: np.random.Generator,
    scenario: str,
    phase: float,
    environment: dict[str, float],
    traits: AnimalTraits,
    targeted_level: float,
    measurement_noise: float,
    management_noise: float,
) -> dict[str, float]:
    env = dict(environment)
    modifiers = _scenario_modifiers(scenario, phase)

    env["thi_norm"] = _clamp01(env["thi_norm"] + modifiers["thi_boost"])
    env["ventilation_quality"] = _clamp01(
        env["ventilation_quality"] - modifiers["ventilation_drop"]
    )
    env["water_status"] = _clamp01(env["water_status"] - modifiers["water_drop"])
    env["feed_delivery_quality"] = _clamp01(
        env["feed_delivery_quality"] - modifiers["feed_drop"]
    )
    env["bedding_quality"] = _clamp01(
        env["bedding_quality"] - modifiers["bedding_drop"]
    )

    exposure = max(0.15, float(targeted_level))

    heat_load = env["thi_norm"] * traits.heat_sensitivity * exposure
    feed_stress = (
        (1.0 - env["feed_delivery_quality"])
        * traits.intake_sensitivity
        * exposure
    )
    water_stress = (1.0 - env["water_status"]) * 0.75 * exposure
    ventilation_stress = (1.0 - env["ventilation_quality"]) * 0.70 * exposure
    bedding_stress = (
        (1.0 - env["bedding_quality"])
        * traits.lameness_sensitivity
        * exposure
    )

    lameness_signal = 0.0
    if scenario == "lameness_cluster":
        lameness_signal = (
            0.85 * phase * traits.lameness_sensitivity * targeted_level
        )

    recovery_pull = 0.0
    if scenario == "post_event_recovery":
        recovery_pull = (
            0.55 * phase * traits.recovery_speed * targeted_level
        )

    stress_core = _clamp01(
        0.32 * heat_load
        + 0.18 * feed_stress
        + 0.16 * water_stress
        + 0.12 * ventilation_stress
        + 0.12 * bedding_stress
        + 0.28 * lameness_signal
        - 0.20 * recovery_pull
        + rng.normal(0, measurement_noise)
    )

    rumination = _clamp01(
        0.78 - 0.62 * stress_core + rng.normal(0, measurement_noise)
    )
    activity = _clamp01(
        0.58
        - 0.25 * heat_load
        - 0.35 * lameness_signal
        + rng.normal(0, measurement_noise)
    )
    locomotion = _clamp01(
        0.82
        - 0.55 * lameness_signal
        - 0.18 * bedding_stress
        + rng.normal(0, measurement_noise)
    )
    feeding = _clamp01(
        0.80
        - 0.45 * feed_stress
        - 0.10 * heat_load
        + rng.normal(0, measurement_noise)
    )
    drinking_pressure = _clamp01(
        0.30
        + 0.45 * heat_load
        + 0.20 * water_stress
        + rng.normal(0, measurement_noise)
    )
    resting_instability = _clamp01(
        0.15
        + 0.40 * heat_load
        + 0.30 * lameness_signal
        + rng.normal(0, measurement_noise)
    )
    respiration = _clamp01(
        0.22
        + 0.58 * heat_load
        + 0.18 * ventilation_stress
        + rng.normal(0, measurement_noise)
    )
    thermal = _clamp01(
        0.20 + 0.65 * heat_load + rng.normal(0, measurement_noise)
    )
    management = _clamp01(
        0.18
        + 0.38 * (1.0 - env["feed_delivery_quality"])
        + 0.30 * (1.0 - env["water_status"])
        + rng.normal(0, management_noise)
    )
    visual = _clamp01(
        0.35 * (1.0 - locomotion)
        + 0.22 * resting_instability
        + 0.20 * respiration
        + 0.15 * management
        + rng.normal(0, 0.02)
    )

    return {
        "thi_norm": env["thi_norm"],
        "ventilation_quality": env["ventilation_quality"],
        "water_status": env["water_status"],
        "feed_delivery_quality": env["feed_delivery_quality"],
        "bedding_quality": env["bedding_quality"],
        "rumination": rumination,
        "activity": activity,
        "locomotion_quality": locomotion,
        "feeding_engagement": feeding,
        "drinking_pressure": drinking_pressure,
        "resting_instability": resting_instability,
        "respiration_load": respiration,
        "thermal_discomfort": thermal,
        "management_disruption": management,
        "visual_anomaly_proxy": visual,
        "stress_core": stress_core,
    }


def animal_risk_score(signals: dict[str, float]) -> float:
    """Historical multimodal fusion weights from the original simulator."""

    score = (
        0.18 * (1.0 - signals["rumination"])
        + 0.14 * (1.0 - signals["activity"])
        + 0.18 * (1.0 - signals["locomotion_quality"])
        + 0.12 * (1.0 - signals["feeding_engagement"])
        + 0.10 * signals["drinking_pressure"]
        + 0.10 * signals["resting_instability"]
        + 0.13 * signals["respiration_load"]
        + 0.12 * signals["thermal_discomfort"]
        + 0.11 * signals["management_disruption"]
        + 0.10 * signals["visual_anomaly_proxy"]
    )
    return _clamp01(score)


def animal_regime(score: float, cfg: SimulationConfig) -> tuple[str, str]:
    if score >= cfg.red_animal_threshold:
        return "collapse-risk", "RED"
    if score >= cfg.yellow_animal_threshold:
        return "transition", "YELLOW"
    return "stable", "GREEN"


def _animal_action(signals: dict[str, float], score: float, cfg: SimulationConfig) -> str:
    actions: list[str] = []

    if signals["thermal_discomfort"] >= 0.55 or signals["respiration_load"] >= 0.55:
        actions.append("cooling_check")
    if signals["water_status"] <= 0.45 or signals["drinking_pressure"] >= 0.55:
        actions.append("water_access_check")
    if (
        signals["feed_delivery_quality"] <= 0.45
        or signals["feeding_engagement"] <= 0.45
    ):
        actions.append("feeding_check")
    if signals["locomotion_quality"] <= 0.45:
        actions.append("locomotion_exam")
    if signals["management_disruption"] >= 0.50:
        actions.append("management_review")

    if score >= cfg.red_animal_threshold:
        actions.insert(0, "on_site_inspection")
    elif score >= cfg.yellow_animal_threshold:
        actions.insert(0, "next_routine_inspection")

    return ";".join(dict.fromkeys(actions)) if actions else "standard_monitoring"


def group_state(group: pd.DataFrame, cfg: SimulationConfig) -> tuple[str, str]:
    red_share = float((group["animal_severity"] == "RED").mean())
    yellow_share = float(
        group["animal_severity"].isin(["YELLOW", "RED"]).mean()
    )
    mean_score = float(group["animal_score"].mean())

    if red_share >= cfg.red_group_share or mean_score >= 0.66:
        return "collapse-risk", "RED"
    if yellow_share >= cfg.yellow_group_share or mean_score >= 0.44:
        return "transition", "YELLOW"
    return "stable", "GREEN"


def unit_state(unit_groups: pd.DataFrame, cfg: SimulationConfig) -> tuple[str, str]:
    red_share = float((unit_groups["group_severity"] == "RED").mean())
    yellow_share = float(
        unit_groups["group_severity"].isin(["YELLOW", "RED"]).mean()
    )
    mean_score = float(unit_groups["group_score"].mean())

    if red_share >= cfg.red_unit_share or mean_score >= 0.65:
        return "collapse-risk", "RED"
    if yellow_share >= cfg.yellow_unit_share or mean_score >= 0.43:
        return "transition", "YELLOW"
    return "stable", "GREEN"


def _group_action(group: pd.DataFrame, severity: str) -> str:
    actions: list[str] = []
    if severity == "RED":
        actions.append("group_intervention")
    elif severity == "YELLOW":
        actions.append("group_monitoring")

    if float(group["thermal_discomfort_mean"].mean()) >= 0.5:
        actions.append("heat_stress_protocol")
    if float(group["locomotion_drop_mean"].mean()) >= 0.3:
        actions.append("mobility_screening")

    actions.append("feed_water_bedding_check")
    return ";".join(dict.fromkeys(actions))


def _unit_action(unit: pd.DataFrame, severity: str) -> str:
    actions: list[str] = []
    if severity == "RED":
        actions.append("unit_emergency_response")
    elif severity == "YELLOW":
        actions.append("unit_level_review")

    if float(unit["group_heat_mean"].mean()) >= 0.5:
        actions.append("climate_system_adjustment")
    if float(unit["group_management_mean"].mean()) >= 0.45:
        actions.append("management_workflow_review")

    actions.append("supervisor_notification")
    return ";".join(dict.fromkeys(actions))


def _simulate_one_run(run_id: int, scenario: str, cfg: SimulationConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_ids = [f"Unit_{i + 1}" for i in range(cfg.n_units)]
    groups = {
        unit: [f"{unit}_Group_{i + 1}" for i in range(cfg.groups_per_unit)]
        for unit in unit_ids
    }

    target_unit = rng.choice(unit_ids)
    target_groups = list(
        rng.choice(
            groups[target_unit],
            size=max(1, min(2, cfg.groups_per_unit)),
            replace=False,
        )
    )

    target_animals: dict[str, set[str]] = {}
    traits: dict[str, AnimalTraits] = {}

    for unit in unit_ids:
        for group in groups[unit]:
            ids = [f"{group}_Animal_{i + 1:02d}" for i in range(cfg.animals_per_group)]
            target_animals[group] = set(
                rng.choice(
                    ids,
                    size=max(3, cfg.animals_per_group // 3),
                    replace=False,
                )
            )
            for animal_id in ids:
                traits[animal_id] = _animal_traits(rng)

    animal_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for hour in range(0, cfg.hours, cfg.time_step_hours):
        phase = _scenario_profile(hour, cfg.hours)
        if scenario == "stable_baseline":
            phase = 0.0

        environment = _baseline_environment(
            rng, hour, cfg.environment_noise
        )

        for unit in unit_ids:
            for group in groups[unit]:
                animal_rows_this_group: list[dict[str, Any]] = []

                for index in range(cfg.animals_per_group):
                    animal_id = f"{group}_Animal_{index + 1:02d}"
                    direct = 0.0

                    if scenario == "post_event_recovery":
                        if unit == target_unit and group in target_groups:
                            direct = 1.0 if animal_id in target_animals[group] else 0.5
                    elif scenario != "stable_baseline" and unit == target_unit:
                        if group in target_groups:
                            direct = 1.0 if animal_id in target_animals[group] else 0.55
                        else:
                            direct = 0.25

                    signals = _simulate_signals(
                        rng=rng,
                        scenario=scenario,
                        phase=phase,
                        environment=environment,
                        traits=traits[animal_id],
                        targeted_level=direct,
                        measurement_noise=cfg.measurement_noise,
                        management_noise=cfg.management_noise,
                    )

                    score = animal_risk_score(signals)
                    regime, severity = animal_regime(score, cfg)

                    row = {
                        "run_id": run_id,
                        "scenario": scenario,
                        "hour": hour,
                        "unit_id": unit,
                        "group_id": group,
                        "animal_id": animal_id,
                        "animal_score": round(score, 4),
                        "animal_regime": regime,
                        "animal_severity": severity,
                        "animal_action": _animal_action(signals, score, cfg),
                        "is_targeted": int(direct >= 0.9),
                        "phase": round(phase, 4),
                        **{k: round(v, 4) for k, v in signals.items()},
                    }
                    animal_rows.append(row)
                    animal_rows_this_group.append(row)

                group_frame = pd.DataFrame(animal_rows_this_group)
                group_regime, group_severity = group_state(group_frame, cfg)
                group_row = {
                    "run_id": run_id,
                    "scenario": scenario,
                    "hour": hour,
                    "unit_id": unit,
                    "group_id": group,
                    "group_score": round(float(group_frame["animal_score"].mean()), 4),
                    "group_regime": group_regime,
                    "group_severity": group_severity,
                    "animals_total": int(len(group_frame)),
                    "animals_red": int((group_frame["animal_severity"] == "RED").sum()),
                    "animals_yellow": int((group_frame["animal_severity"] == "YELLOW").sum()),
                    "animals_green": int((group_frame["animal_severity"] == "GREEN").sum()),
                    "thermal_discomfort_mean": round(float(group_frame["thermal_discomfort"].mean()), 4),
                    "locomotion_drop_mean": round(float((1.0 - group_frame["locomotion_quality"]).mean()), 4),
                    "group_heat_mean": round(float(group_frame["thermal_discomfort"].mean()), 4),
                    "group_management_mean": round(float(group_frame["management_disruption"].mean()), 4),
                }
                group_row["group_action"] = _group_action(
                    pd.DataFrame([group_row]), group_severity
                )
                group_rows.append(group_row)

            groups_frame = pd.DataFrame(
                [row for row in group_rows if row["unit_id"] == unit and row["hour"] == hour and row["run_id"] == run_id and row["scenario"] == scenario]
            )
            unit_regime, unit_severity = unit_state(groups_frame, cfg)

            unit_row = {
                "run_id": run_id,
                "scenario": scenario,
                "hour": hour,
                "unit_id": unit,
                "unit_score": round(float(groups_frame["group_score"].mean()), 4),
                "unit_regime": unit_regime,
                "unit_severity": unit_severity,
                "groups_total": int(len(groups_frame)),
                "groups_red": int((groups_frame["group_severity"] == "RED").sum()),
                "groups_yellow": int((groups_frame["group_severity"] == "YELLOW").sum()),
                "groups_green": int((groups_frame["group_severity"] == "GREEN").sum()),
                "group_heat_mean": round(float(groups_frame["group_heat_mean"].mean()), 4),
                "group_management_mean": round(float(groups_frame["group_management_mean"].mean()), 4),
            }
            unit_row["unit_action"] = _unit_action(
                pd.DataFrame([unit_row]), unit_severity
            )
            unit_rows.append(unit_row)

    return (
        pd.DataFrame(animal_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(unit_rows),
    )


def _truth_for_scenario(scenario: str, hour: int, hours: int) -> str:
    if scenario == "stable_baseline":
        return "GREEN"

    start = int(hours * 0.35)
    peak = int(hours * 0.50)
    end = int(hours * 0.78)

    if hour < start:
        return "GREEN"
    if hour < peak:
        return "YELLOW"
    if hour < end:
        return "YELLOW" if scenario == "post_event_recovery" else "RED"
    return "WATCH"


def _metrics(predicted: pd.Series, truth: pd.Series) -> dict[str, float | int]:
    pred = predicted.astype(str).str.upper().map(SEVERITY_RANK).fillna(0).astype(int)
    actual = truth.astype(str).str.upper().map(SEVERITY_RANK).fillna(0).astype(int)

    exact = float((pred == actual).mean())
    pred_concern = pred >= 2
    truth_concern = actual >= 2

    tp = int((pred_concern & truth_concern).sum())
    fp = int((pred_concern & ~truth_concern).sum())
    fn = int((~pred_concern & truth_concern).sum())
    tn = int((~pred_concern & ~truth_concern).sum())

    return {
        "exact_match_rate": exact,
        "precision": tp / (tp + fp + 1e-12),
        "recall": tp / (tp + fn + 1e-12),
        "specificity": tn / (tn + fp + 1e-12),
        "accuracy": (tp + tn) / max(len(pred), 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "red_rate": float((pred == 3).mean()),
        "concern_rate": float((pred >= 2).mean()),
    }


def simulate(
    config: SimulationConfig | None = None,
    *,
    scenarios: tuple[str, ...] = SCENARIOS,
) -> dict[str, Any]:
    """Run one or more multimodal Monte Carlo simulations.

    Returns DataFrames for animal, group and unit trajectories plus a
    scenario-level evaluation table.
    """

    cfg = config or SimulationConfig()
    if cfg.monte_carlo < 1:
        raise ValueError("monte_carlo must be >= 1")
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")

    root_rng = np.random.default_rng(cfg.seed)
    animals: list[pd.DataFrame] = []
    groups: list[pd.DataFrame] = []
    units: list[pd.DataFrame] = []

    for run_id in range(cfg.monte_carlo):
        run_seed = int(root_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(run_seed)

        for scenario in scenarios:
            a, g, u = _simulate_one_run(run_id, scenario, cfg, rng)
            animals.append(a)
            groups.append(g)
            units.append(u)

    animal_df = pd.concat(animals, ignore_index=True)
    group_df = pd.concat(groups, ignore_index=True)
    unit_df = pd.concat(units, ignore_index=True)

    scenario_rows = []
    for scenario, frame in unit_df.groupby("scenario"):
        truth = pd.Series(
            [
                _truth_for_scenario(scenario, int(hour), cfg.hours)
                for hour in frame["hour"]
            ],
            index=frame.index,
        )
        row = {"scenario": scenario}
        row.update(_metrics(frame["unit_severity"], truth))
        scenario_rows.append(row)

    scenario_metrics = pd.DataFrame(scenario_rows)

    return {
        "animals": animal_df,
        "groups": group_df,
        "units": unit_df,
        "scenario_metrics": scenario_metrics,
        "config": cfg,
    }
