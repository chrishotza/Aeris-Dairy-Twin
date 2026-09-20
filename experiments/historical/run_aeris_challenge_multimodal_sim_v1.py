import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# CHALLENGE-ORIENTED MULTIMODAL SIMULATOR
# AERIS integrated simulation for:
# - behavioral
# - biomechanical
# - physiological
# - environmental
# - management
# Outputs:
# - animal/group/unit trajectories
# - alert feed
# - incident summaries
# - lead time / alert quality metrics
# - scenario coverage and robustness summary
# =========================================================

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ROOT = Path(".").resolve()
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# HEAVY KNOBS
# ---------------------------------------------------------
RANDOM_SEED = 2042

N_MONTE_CARLO = 60          # subilo si querés más pesado
N_UNITS = 4
GROUPS_PER_UNIT = 4
ANIMALS_PER_GROUP = 14
HOURS = 96                  # 4 días
TIME_STEP_HOURS = 1

# noise / perturbation
MEAS_NOISE = 0.035
ENV_NOISE = 0.04
MANAGEMENT_NOISE = 0.03

# engine thresholds
YELLOW_ANIMAL_THRESH = 0.42
RED_ANIMAL_THRESH = 0.70

YELLOW_GROUP_SHARE = 0.20
RED_GROUP_SHARE = 0.33

YELLOW_UNIT_SHARE = 0.15
RED_UNIT_SHARE = 0.25

# alert persistence
MIN_ALERT_DURATION = 2

# ---------------------------------------------------------
# FIXED SEVERITY MAPS
# ---------------------------------------------------------
SEV_RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
RANK_SEV = {v: k for k, v in SEV_RANK.items()}

# ---------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------
SCENARIOS = [
    "stable_baseline",
    "heat_stress_wave",
    "lameness_cluster",
    "feeding_disruption",
    "water_system_issue",
    "ventilation_failure",
    "post_event_recovery",
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def zsafe(x, center, scale):
    if scale == 0:
        return 0.0
    return (x - center) / scale

def rolling_mean(arr, w):
    if w <= 1:
        return arr.copy()
    s = pd.Series(arr)
    return s.rolling(w, min_periods=1).mean().to_numpy()

# ---------------------------------------------------------
# STRUCTURES
# ---------------------------------------------------------
@dataclass
class SimConfig:
    monte_carlo: int = N_MONTE_CARLO
    n_units: int = N_UNITS
    groups_per_unit: int = GROUPS_PER_UNIT
    animals_per_group: int = ANIMALS_PER_GROUP
    hours: int = HOURS
    time_step_hours: int = TIME_STEP_HOURS

# ---------------------------------------------------------
# SCENARIO SCHEDULE
# ---------------------------------------------------------
def build_scenario_schedule(hours: int, scenario_name: str):
    """
    Returns a dict describing event windows and intensities.
    """
    sched = {
        "scenario": scenario_name,
        "event_start": int(hours * 0.35),
        "event_peak": int(hours * 0.55),
        "event_end": int(hours * 0.78),
        "target_units": [],
        "target_groups": [],
        "target_animals": [],
    }
    return sched

def scenario_profile(hour, sched):
    start = sched["event_start"]
    peak = sched["event_peak"]
    end = sched["event_end"]

    if hour < start:
        return 0.0
    if start <= hour <= peak:
        return (hour - start) / max(1, peak - start)
    if peak < hour <= end:
        return 1.0 - 0.70 * ((hour - peak) / max(1, end - peak))
    return 0.0

# ---------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------
def baseline_environment(hour):
    # coarse diurnal cycle
    day_phase = math.sin((2 * math.pi * hour) / 24.0)
    temp = 23 + 6 * day_phase + np.random.normal(0, 0.6)
    humidity = 58 - 8 * day_phase + np.random.normal(0, 1.0)
    ventilation_quality = clamp01(0.82 + np.random.normal(0, 0.04))
    water_status = clamp01(0.92 + np.random.normal(0, 0.03))
    feed_delivery_quality = clamp01(0.90 + np.random.normal(0, 0.03))
    bedding_quality = clamp01(0.86 + np.random.normal(0, 0.03))

    # simplified THI-like proxy normalized
    thi_raw = 0.7 * temp + 0.3 * (humidity / 2.0)
    thi_norm = clamp01((thi_raw - 18.0) / 20.0)

    return {
        "temp_c": temp,
        "humidity_pct": humidity,
        "thi_norm": thi_norm,
        "ventilation_quality": ventilation_quality,
        "water_status": water_status,
        "feed_delivery_quality": feed_delivery_quality,
        "bedding_quality": bedding_quality,
    }

def scenario_environment_modifiers(scenario_name, phase):
    # phase in [0,1] approximately
    mod = {
        "thi_boost": 0.0,
        "ventilation_drop": 0.0,
        "water_drop": 0.0,
        "feed_drop": 0.0,
        "bedding_drop": 0.0,
    }

    if scenario_name == "heat_stress_wave":
        mod["thi_boost"] = 0.40 * phase
        mod["ventilation_drop"] = 0.10 * phase
    elif scenario_name == "feeding_disruption":
        mod["feed_drop"] = 0.50 * phase
    elif scenario_name == "water_system_issue":
        mod["water_drop"] = 0.55 * phase
    elif scenario_name == "ventilation_failure":
        mod["ventilation_drop"] = 0.60 * phase
        mod["thi_boost"] = 0.20 * phase
    elif scenario_name == "lameness_cluster":
        mod["bedding_drop"] = 0.20 * phase
    elif scenario_name == "post_event_recovery":
        mod["bedding_drop"] = 0.15 * phase
        mod["feed_drop"] = 0.10 * phase

    return mod

def animal_baseline_traits():
    return {
        "resilience": clamp01(np.random.normal(0.62, 0.10)),
        "heat_sensitivity": clamp01(np.random.normal(0.55, 0.12)),
        "lameness_sensitivity": clamp01(np.random.normal(0.50, 0.15)),
        "intake_sensitivity": clamp01(np.random.normal(0.52, 0.13)),
        "recovery_speed": clamp01(np.random.normal(0.58, 0.10)),
    }

def simulate_animal_signals(hour, base_env, scenario_name, phase, traits, targeted_level):
    """
    targeted_level:
      0.0 = not targeted
      0.5 = weak exposure
      1.0 = direct exposure
    """

    env = base_env.copy()
    mod = scenario_environment_modifiers(scenario_name, phase)

    env["thi_norm"] = clamp01(env["thi_norm"] + mod["thi_boost"])
    env["ventilation_quality"] = clamp01(env["ventilation_quality"] - mod["ventilation_drop"])
    env["water_status"] = clamp01(env["water_status"] - mod["water_drop"])
    env["feed_delivery_quality"] = clamp01(env["feed_delivery_quality"] - mod["feed_drop"])
    env["bedding_quality"] = clamp01(env["bedding_quality"] - mod["bedding_drop"])

    # scenario exposures
    heat_load = env["thi_norm"] * traits["heat_sensitivity"] * max(0.15, targeted_level)
    feed_stress = (1.0 - env["feed_delivery_quality"]) * traits["intake_sensitivity"] * max(0.15, targeted_level)
    water_stress = (1.0 - env["water_status"]) * 0.75 * max(0.15, targeted_level)
    ventilation_stress = (1.0 - env["ventilation_quality"]) * 0.70 * max(0.15, targeted_level)
    bedding_stress = (1.0 - env["bedding_quality"]) * traits["lameness_sensitivity"] * max(0.15, targeted_level)

    lameness_signal = 0.0
    if scenario_name == "lameness_cluster":
        lameness_signal = 0.85 * phase * traits["lameness_sensitivity"] * targeted_level

    # recovery effect
    recovery_pull = 0.0
    if scenario_name == "post_event_recovery":
        recovery_pull = 0.55 * phase * traits["recovery_speed"] * targeted_level

    stress_core = (
        0.32 * heat_load +
        0.18 * feed_stress +
        0.16 * water_stress +
        0.12 * ventilation_stress +
        0.12 * bedding_stress +
        0.28 * lameness_signal -
        0.20 * recovery_pull
    )

    stress_core = clamp01(stress_core + np.random.normal(0, MEAS_NOISE))

    # observable signals
    rumination = clamp01(0.78 - 0.62 * stress_core + np.random.normal(0, MEAS_NOISE))
    activity = clamp01(0.58 - 0.25 * heat_load - 0.35 * lameness_signal + np.random.normal(0, MEAS_NOISE))
    locomotion_quality = clamp01(0.82 - 0.55 * lameness_signal - 0.18 * bedding_stress + np.random.normal(0, MEAS_NOISE))
    feeding_engagement = clamp01(0.80 - 0.45 * feed_stress - 0.10 * heat_load + np.random.normal(0, MEAS_NOISE))
    drinking_pressure = clamp01(0.30 + 0.45 * heat_load + 0.20 * water_stress + np.random.normal(0, MEAS_NOISE))
    resting_instability = clamp01(0.15 + 0.40 * heat_load + 0.30 * lameness_signal + np.random.normal(0, MEAS_NOISE))
    respiration_load = clamp01(0.22 + 0.58 * heat_load + 0.18 * ventilation_stress + np.random.normal(0, MEAS_NOISE))
    thermal_discomfort = clamp01(0.20 + 0.65 * heat_load + np.random.normal(0, MEAS_NOISE))
    management_disruption = clamp01(
        0.18
        + 0.38 * (1.0 - env["feed_delivery_quality"])
        + 0.30 * (1.0 - env["water_status"])
        + np.random.normal(0, MANAGEMENT_NOISE)
    )

    # visual proxy, aligned with your current visual submodule
    visual_anomaly_proxy = clamp01(
        0.35 * (1.0 - locomotion_quality) +
        0.22 * resting_instability +
        0.20 * respiration_load +
        0.15 * management_disruption +
        np.random.normal(0, 0.02)
    )

    return {
        "thi_norm": env["thi_norm"],
        "ventilation_quality": env["ventilation_quality"],
        "water_status": env["water_status"],
        "feed_delivery_quality": env["feed_delivery_quality"],
        "bedding_quality": env["bedding_quality"],
        "rumination": rumination,
        "activity": activity,
        "locomotion_quality": locomotion_quality,
        "feeding_engagement": feeding_engagement,
        "drinking_pressure": drinking_pressure,
        "resting_instability": resting_instability,
        "respiration_load": respiration_load,
        "thermal_discomfort": thermal_discomfort,
        "management_disruption": management_disruption,
        "visual_anomaly_proxy": visual_anomaly_proxy,
        "stress_core": stress_core,
    }

# ---------------------------------------------------------
# ENGINE: animal -> group -> unit
# ---------------------------------------------------------
def animal_risk_score(sig):
    # challenge-oriented multimodal fusion
    score = (
        0.18 * (1.0 - sig["rumination"]) +
        0.14 * (1.0 - sig["activity"]) +
        0.18 * (1.0 - sig["locomotion_quality"]) +
        0.12 * (1.0 - sig["feeding_engagement"]) +
        0.10 * sig["drinking_pressure"] +
        0.10 * sig["resting_instability"] +
        0.13 * sig["respiration_load"] +
        0.12 * sig["thermal_discomfort"] +
        0.11 * sig["management_disruption"] +
        0.10 * sig["visual_anomaly_proxy"]
    )

    return clamp01(score)

def animal_regime_from_score(score):
    if score >= RED_ANIMAL_THRESH:
        return "collapse-risk", "RED"
    if score >= YELLOW_ANIMAL_THRESH:
        return "transition", "YELLOW"
    return "stable", "GREEN"

def recommended_action_animal(sig, score):
    actions = []
    if sig["thermal_discomfort"] >= 0.55 or sig["respiration_load"] >= 0.55:
        actions.append("cooling_check")
    if sig["water_status"] <= 0.45 or sig["drinking_pressure"] >= 0.55:
        actions.append("water_access_check")
    if sig["feed_delivery_quality"] <= 0.45 or sig["feeding_engagement"] <= 0.45:
        actions.append("feeding_check")
    if sig["locomotion_quality"] <= 0.45:
        actions.append("locomotion_exam")
    if sig["management_disruption"] >= 0.50:
        actions.append("management_review")

    if score >= RED_ANIMAL_THRESH and "on_site_inspection" not in actions:
        actions.insert(0, "on_site_inspection")
    elif score >= YELLOW_ANIMAL_THRESH and "next_routine_inspection" not in actions:
        actions.insert(0, "next_routine_inspection")

    return ";".join(actions) if actions else "standard_monitoring"

def group_state(df_group):
    red_share = (df_group["animal_severity"] == "RED").mean()
    yellow_share = ((df_group["animal_severity"] == "YELLOW") | (df_group["animal_severity"] == "RED")).mean()
    mean_score = df_group["animal_score"].mean()

    if red_share >= RED_GROUP_SHARE or mean_score >= 0.66:
        return "collapse-risk", "RED"
    if yellow_share >= YELLOW_GROUP_SHARE or mean_score >= 0.44:
        return "transition", "YELLOW"
    return "stable", "GREEN"

def unit_state(df_unit):
    red_share = (df_unit["group_severity"] == "RED").mean()
    yellow_share = ((df_unit["group_severity"] == "YELLOW") | (df_unit["group_severity"] == "RED")).mean()
    mean_score = df_unit["group_score"].mean()

    if red_share >= RED_UNIT_SHARE or mean_score >= 0.65:
        return "collapse-risk", "RED"
    if yellow_share >= YELLOW_UNIT_SHARE or mean_score >= 0.43:
        return "transition", "YELLOW"
    return "stable", "GREEN"

def recommended_action_group(df_group, severity):
    hot = df_group["thermal_discomfort_mean"].mean() if "thermal_discomfort_mean" in df_group.columns else 0.0
    loco = df_group["locomotion_drop_mean"].mean() if "locomotion_drop_mean" in df_group.columns else 0.0

    acts = []
    if severity == "RED":
        acts.append("group_intervention")
    elif severity == "YELLOW":
        acts.append("group_monitoring")

    if hot >= 0.5:
        acts.append("heat_stress_protocol")
    if loco >= 0.3:
        acts.append("mobility_screening")
    acts.append("feed_water_bedding_check")

    return ";".join(dict.fromkeys(acts))

def recommended_action_unit(df_unit, severity):
    acts = []
    if severity == "RED":
        acts.append("unit_emergency_response")
    elif severity == "YELLOW":
        acts.append("unit_level_review")

    if df_unit["group_heat_mean"].mean() >= 0.5:
        acts.append("climate_system_adjustment")
    if df_unit["group_management_mean"].mean() >= 0.45:
        acts.append("management_workflow_review")

    acts.append("supervisor_notification")
    return ";".join(dict.fromkeys(acts))

# ---------------------------------------------------------
# INCIDENT AND ALERT LOGIC
# ---------------------------------------------------------
def severity_changed(prev, cur):
    return SEV_RANK.get(cur, 0) > SEV_RANK.get(prev, 0)

def build_alert_feed(df_states):
    alerts = []
    for level_col, id_col, sev_col, reg_col, act_col in [
        ("animal", "animal_id", "animal_severity", "animal_regime", "animal_action"),
        ("group", "group_id", "group_severity", "group_regime", "group_action"),
        ("unit", "unit_id", "unit_severity", "unit_regime", "unit_action"),
    ]:
        subset = df_states.sort_values([id_col, "hour"]).copy()
        for ent_id, g in subset.groupby(id_col):
            prev = "GREEN"
            alert_open_hour = None
            for _, row in g.iterrows():
                cur = row[sev_col]
                if severity_changed(prev, cur):
                    alerts.append({
                        "entity_level": level_col,
                        "entity_id": ent_id,
                        "hour": int(row["hour"]),
                        "severity": cur,
                        "regime": row[reg_col],
                        "recommended_action": row[act_col],
                        "run_id": int(row["run_id"]),
                        "scenario": row["scenario"],
                    })
                    alert_open_hour = int(row["hour"])
                prev = cur
    alerts_df = pd.DataFrame(alerts)
    if len(alerts_df):
        alerts_df = alerts_df.sort_values(["run_id", "entity_level", "entity_id", "hour"]).reset_index(drop=True)
        alerts_df["alert_id"] = [f"SIMALT_{i+1:06d}" for i in range(len(alerts_df))]
    return alerts_df

# ---------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------
def simulate_run(run_id: int, scenario_name: str, cfg: SimConfig):
    rows_anim = []
    rows_group = []
    rows_unit = []

    sched = build_scenario_schedule(cfg.hours, scenario_name)

    # pick target structure
    unit_ids = [f"Unit_{u+1}" for u in range(cfg.n_units)]
    group_ids_by_unit = {u: [f"{u}_Group_{g+1}" for g in range(cfg.groups_per_unit)] for u in unit_ids}

    target_unit = random.choice(unit_ids)
    target_groups = random.sample(group_ids_by_unit[target_unit], k=max(1, min(2, cfg.groups_per_unit)))
    target_animals = {}

    for g in target_groups:
        animals = [f"{g}_Animal_{i+1:02d}" for i in range(cfg.animals_per_group)]
        target_animals[g] = set(random.sample(animals, k=max(3, cfg.animals_per_group // 3)))

    traits_map = {}
    for u in unit_ids:
        for g in group_ids_by_unit[u]:
            for i in range(cfg.animals_per_group):
                aid = f"{g}_Animal_{i+1:02d}"
                traits_map[aid] = animal_baseline_traits()

    for hour in range(cfg.hours):
        phase = scenario_profile(hour, sched)
        env = baseline_environment(hour)

        if scenario_name == "stable_baseline":
            phase = 0.0

        group_cache = []
        unit_cache = []

        for u in unit_ids:
            for g in group_ids_by_unit[u]:
                animal_cache = []

                for i in range(cfg.animals_per_group):
                    aid = f"{g}_Animal_{i+1:02d}"
                    direct = 0.0

                    if scenario_name == "stable_baseline":
                        direct = 0.0
                    elif scenario_name == "post_event_recovery":
                        if u == target_unit and g in target_groups:
                            direct = 1.0 if aid in target_animals[g] else 0.5
                    else:
                        if u == target_unit:
                            if g in target_groups:
                                direct = 1.0 if aid in target_animals[g] else 0.55
                            else:
                                direct = 0.25

                    sig = simulate_animal_signals(hour, env, scenario_name, phase, traits_map[aid], direct)
                    score = animal_risk_score(sig)
                    regime, sev = animal_regime_from_score(score)
                    action = recommended_action_animal(sig, score)

                    row = {
                        "run_id": run_id,
                        "scenario": scenario_name,
                        "hour": hour,
                        "unit_id": u,
                        "group_id": g,
                        "animal_id": aid,
                        "animal_score": round(score, 4),
                        "animal_regime": regime,
                        "animal_severity": sev,
                        "animal_action": action,
                        "is_targeted": int(direct >= 0.9),
                        "phase": round(phase, 4),
                    }
                    row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in sig.items()})
                    rows_anim.append(row)
                    animal_cache.append(row)

                df_g_anim = pd.DataFrame(animal_cache)
                g_reg, g_sev = group_state(df_g_anim)
                g_score = float(df_g_anim["animal_score"].mean())

                group_row = {
                    "run_id": run_id,
                    "scenario": scenario_name,
                    "hour": hour,
                    "unit_id": u,
                    "group_id": g,
                    "group_score": round(g_score, 4),
                    "group_regime": g_reg,
                    "group_severity": g_sev,
                    "animals_total": len(df_g_anim),
                    "animals_red": int((df_g_anim["animal_severity"] == "RED").sum()),
                    "animals_yellow": int((df_g_anim["animal_severity"] == "YELLOW").sum()),
                    "animals_green": int((df_g_anim["animal_severity"] == "GREEN").sum()),
                    "thermal_discomfort_mean": round(float(df_g_anim["thermal_discomfort"].mean()), 4),
                    "locomotion_drop_mean": round(float((1.0 - df_g_anim["locomotion_quality"]).mean()), 4),
                    "group_heat_mean": round(float(df_g_anim["thermal_discomfort"].mean()), 4),
                    "group_management_mean": round(float(df_g_anim["management_disruption"].mean()), 4),
                }
                group_row["group_action"] = recommended_action_group(pd.DataFrame([group_row]), g_sev)
                rows_group.append(group_row)
                group_cache.append(group_row)

            df_u_group = pd.DataFrame([r for r in group_cache if r["unit_id"] == u])
            u_reg, u_sev = unit_state(df_u_group)
            u_score = float(df_u_group["group_score"].mean())

            unit_row = {
                "run_id": run_id,
                "scenario": scenario_name,
                "hour": hour,
                "unit_id": u,
                "unit_score": round(u_score, 4),
                "unit_regime": u_reg,
                "unit_severity": u_sev,
                "groups_total": len(df_u_group),
                "groups_red": int((df_u_group["group_severity"] == "RED").sum()),
                "groups_yellow": int((df_u_group["group_severity"] == "YELLOW").sum()),
                "groups_green": int((df_u_group["group_severity"] == "GREEN").sum()),
                "group_heat_mean": round(float(df_u_group["group_heat_mean"].mean()), 4),
                "group_management_mean": round(float(df_u_group["group_management_mean"].mean()), 4),
            }
            unit_row["unit_action"] = recommended_action_unit(df_u_group, u_sev)
            rows_unit.append(unit_row)
            unit_cache.append(unit_row)

    return (
        pd.DataFrame(rows_anim),
        pd.DataFrame(rows_group),
        pd.DataFrame(rows_unit),
        {
            "run_id": run_id,
            "scenario": scenario_name,
            "target_unit": target_unit,
            "target_groups": ";".join(target_groups),
        },
    )

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
def derive_ground_truth_level(scenario_name, hour, sched_like=None):
    # approximate challenge-oriented ground truth
    # stable_baseline -> always GREEN
    if scenario_name == "stable_baseline":
        return "GREEN"
    if hour < int(HOURS * 0.35):
        return "GREEN"
    if int(HOURS * 0.35) <= hour < int(HOURS * 0.50):
        return "YELLOW"
    if int(HOURS * 0.50) <= hour < int(HOURS * 0.78):
        return "RED" if scenario_name not in ["post_event_recovery"] else "YELLOW"
    if scenario_name == "post_event_recovery":
        return "WATCH"
    return "WATCH"

def alert_quality_metrics(unit_df):
    # evaluate at unit level for challenge-facing summary
    eval_rows = []
    for (run_id, scenario), g in unit_df.groupby(["run_id", "scenario"]):
        g = g.sort_values("hour").reset_index(drop=True)

        y_true = [derive_ground_truth_level(scenario, int(h)) for h in g["hour"]]
        y_pred = g["unit_severity"].astype(str).tolist()

        true_concern = np.array([1 if x in ["YELLOW", "RED"] else 0 for x in y_true], dtype=int)
        pred_concern = np.array([1 if x in ["YELLOW", "RED"] else 0 for x in y_pred], dtype=int)

        tp = int(((pred_concern == 1) & (true_concern == 1)).sum())
        fp = int(((pred_concern == 1) & (true_concern == 0)).sum())
        fn = int(((pred_concern == 0) & (true_concern == 1)).sum())
        tn = int(((pred_concern == 0) & (true_concern == 0)).sum())

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        accuracy = (tp + tn) / max(len(g), 1)

        # lead time to first YELLOW and first RED
        try:
            first_true_red = int(g.loc[[i for i, x in enumerate(y_true) if x == "RED"][0], "hour"])
        except IndexError:
            first_true_red = None

        pred_yellow_hours = g.loc[g["unit_severity"].isin(["YELLOW", "RED"]), "hour"].tolist()
        pred_red_hours = g.loc[g["unit_severity"].eq("RED"), "hour"].tolist()

        lead_yellow = None
        lead_red = None
        if first_true_red is not None and pred_yellow_hours:
            lead_yellow = first_true_red - int(pred_yellow_hours[0])
        if first_true_red is not None and pred_red_hours:
            lead_red = first_true_red - int(pred_red_hours[0])

        eval_rows.append({
            "run_id": run_id,
            "scenario": scenario,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4),
            "lead_time_yellow": lead_yellow,
            "lead_time_red": lead_red,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(eval_rows)

def scenario_summary(animal_df, group_df, unit_df, alerts_df, eval_df):
    rows = []
    for scen in SCENARIOS:
        ua = unit_df[unit_df["scenario"] == scen]
        aa = animal_df[animal_df["scenario"] == scen]
        al = alerts_df[alerts_df["scenario"] == scen] if len(alerts_df) else pd.DataFrame()
        ev = eval_df[eval_df["scenario"] == scen] if len(eval_df) else pd.DataFrame()

        rows.append({
            "scenario": scen,
            "animal_mean_score": round(float(aa["animal_score"].mean()), 4) if len(aa) else None,
            "group_mean_score": round(float(group_df[group_df["scenario"] == scen]["group_score"].mean()), 4) if len(group_df) else None,
            "unit_mean_score": round(float(ua["unit_score"].mean()), 4) if len(ua) else None,
            "unit_red_rate": round(float((ua["unit_severity"] == "RED").mean()), 4) if len(ua) else None,
            "unit_yellow_rate": round(float((ua["unit_severity"] == "YELLOW").mean()), 4) if len(ua) else None,
            "alerts_total": int(len(al)),
            "alerts_red": int((al["severity"] == "RED").sum()) if len(al) else 0,
            "alerts_yellow": int((al["severity"] == "YELLOW").sum()) if len(al) else 0,
            "precision_mean": round(float(ev["precision"].mean()), 4) if len(ev) else None,
            "recall_mean": round(float(ev["recall"].mean()), 4) if len(ev) else None,
            "specificity_mean": round(float(ev["specificity"].mean()), 4) if len(ev) else None,
            "accuracy_mean": round(float(ev["accuracy"].mean()), 4) if len(ev) else None,
            "lead_time_yellow_mean": round(float(pd.to_numeric(ev["lead_time_yellow"], errors="coerce").dropna().mean()), 4) if len(ev) and pd.to_numeric(ev["lead_time_yellow"], errors="coerce").dropna().size else None,
            "lead_time_red_mean": round(float(pd.to_numeric(ev["lead_time_red"], errors="coerce").dropna().mean()), 4) if len(ev) and pd.to_numeric(ev["lead_time_red"], errors="coerce").dropna().size else None,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    set_seed(RANDOM_SEED)
    cfg = SimConfig()

    t0 = time.time()

    all_anim = []
    all_group = []
    all_unit = []
    run_meta = []

    print("\n=== AERIS CHALLENGE MULTIMODAL SIM V1 ===")
    print(f"Monte Carlo runs  : {cfg.monte_carlo}")
    print(f"Units             : {cfg.n_units}")
    print(f"Groups per unit   : {cfg.groups_per_unit}")
    print(f"Animals per group : {cfg.animals_per_group}")
    print(f"Hours per run     : {cfg.hours}")
    print("")

    for run_id in range(cfg.monte_carlo):
        scenario_name = SCENARIOS[run_id % len(SCENARIOS)]
        anim_df, group_df, unit_df, meta = simulate_run(run_id, scenario_name, cfg)

        all_anim.append(anim_df)
        all_group.append(group_df)
        all_unit.append(unit_df)
        run_meta.append(meta)

        if (run_id + 1) % 10 == 0 or (run_id + 1) == cfg.monte_carlo:
            print(f"Completed {run_id + 1}/{cfg.monte_carlo} runs")

    animal_df = pd.concat(all_anim, ignore_index=True)
    group_df = pd.concat(all_group, ignore_index=True)
    unit_df = pd.concat(all_unit, ignore_index=True)
    run_meta_df = pd.DataFrame(run_meta)

    states_flat = (
        animal_df[["run_id","scenario","hour","animal_id","animal_severity","animal_regime","animal_action"]]
        .rename(columns={"animal_id": "entity_id"})
        .assign(entity_level="animal")
    )

    alerts_df = build_alert_feed(
        animal_df.merge(
            group_df[["run_id","scenario","hour","group_id","group_severity","group_regime","group_action"]],
            on=["run_id","scenario","hour","group_id"],
            how="left"
        ).merge(
            unit_df[["run_id","scenario","hour","unit_id","unit_severity","unit_regime","unit_action"]],
            on=["run_id","scenario","hour","unit_id"],
            how="left"
        )
    )

    eval_df = alert_quality_metrics(unit_df)
    scen_df = scenario_summary(animal_df, group_df, unit_df, alerts_df, eval_df)

    # robust high-level summary
    global_summary = {
        "rows_animals": int(len(animal_df)),
        "rows_groups": int(len(group_df)),
        "rows_units": int(len(unit_df)),
        "alerts_total": int(len(alerts_df)),
        "alerts_red": int((alerts_df["severity"] == "RED").sum()) if len(alerts_df) else 0,
        "alerts_yellow": int((alerts_df["severity"] == "YELLOW").sum()) if len(alerts_df) else 0,
        "mean_precision_unit": round(float(eval_df["precision"].mean()), 4),
        "mean_recall_unit": round(float(eval_df["recall"].mean()), 4),
        "mean_specificity_unit": round(float(eval_df["specificity"].mean()), 4),
        "mean_accuracy_unit": round(float(eval_df["accuracy"].mean()), 4),
        "mean_lead_time_yellow": round(float(pd.to_numeric(eval_df["lead_time_yellow"], errors="coerce").dropna().mean()), 4) if pd.to_numeric(eval_df["lead_time_yellow"], errors="coerce").dropna().size else None,
        "mean_lead_time_red": round(float(pd.to_numeric(eval_df["lead_time_red"], errors="coerce").dropna().mean()), 4) if pd.to_numeric(eval_df["lead_time_red"], errors="coerce").dropna().size else None,
        "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
    }

    # save
    animal_out = OUT_DIR / "animal_states_challenge_sim_v1.csv"
    group_out = OUT_DIR / "group_states_challenge_sim_v1.csv"
    unit_out = OUT_DIR / "unit_states_challenge_sim_v1.csv"
    alerts_out = OUT_DIR / "alert_feed_challenge_sim_v1.csv"
    eval_out = OUT_DIR / "unit_alert_quality_challenge_sim_v1.csv"
    scen_out = OUT_DIR / "scenario_summary_challenge_sim_v1.csv"
    meta_out = OUT_DIR / "run_meta_challenge_sim_v1.csv"
    json_out = OUT_DIR / "challenge_sim_global_summary_v1.json"
    txt_out = OUT_DIR / "challenge_sim_summary_v1.txt"

    animal_df.to_csv(animal_out, index=False)
    group_df.to_csv(group_out, index=False)
    unit_df.to_csv(unit_out, index=False)
    alerts_df.to_csv(alerts_out, index=False)
    eval_df.to_csv(eval_out, index=False)
    scen_df.to_csv(scen_out, index=False)
    run_meta_df.to_csv(meta_out, index=False)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    lines = []
    lines.append("AERIS CHALLENGE MULTIMODAL SIM V1")
    lines.append("=================================")
    lines.append("")
    lines.append("Global summary:")
    for k, v in global_summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Scenario summary:")
    lines.append(scen_df.to_string(index=False))
    txt_out.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Animal rows : {len(animal_df)}")
    print(f"Group rows  : {len(group_df)}")
    print(f"Unit rows   : {len(unit_df)}")
    print(f"Alerts      : {len(alerts_df)}")
    print("\nGlobal summary:")
    for k, v in global_summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved:")
    print(f"- {animal_out}")
    print(f"- {group_out}")
    print(f"- {unit_out}")
    print(f"- {alerts_out}")
    print(f"- {eval_out}")
    print(f"- {scen_out}")
    print(f"- {meta_out}")
    print(f"- {json_out}")
    print(f"- {txt_out}")

if __name__ == "__main__":
    main()
