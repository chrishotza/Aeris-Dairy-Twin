import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2031)

root = Path(".")
out_dir = root / "07_assets" / "data_placeholders"

# =========================================================
# Config
# =========================================================

timestamps = pd.date_range("2026-04-15 00:00:00", periods=36, freq="2h")
groups = {
    "Pen_B12": ["0971", "1842", "1101", "1102", "1103", "1104", "1105", "1106"],
    "Pen_C04": ["2201", "2202", "2203", "2204", "2205", "2206", "2207", "2208"],
    "Pen_D02": ["3301", "3302", "3303", "3304", "3305", "3306", "3307", "3308"],
}

animal_profiles = {
    "0971": "collapse_recover",
    "1842": "collapse_recover",
    "1101": "transition",
    "1102": "stable",
    "1103": "stable",
    "1104": "late_transition",
    "1105": "stable",
    "1106": "relapse",

    "2201": "transition",
    "2202": "transition",
    "2203": "collapse_recover",
    "2204": "stable",
    "2205": "stable",
    "2206": "late_transition",
    "2207": "stable",
    "2208": "stable",

    "3301": "stable",
    "3302": "stable",
    "3303": "stable",
    "3304": "late_transition",
    "3305": "stable",
    "3306": "stable",
    "3307": "stable",
    "3308": "stable",
}

# =========================================================
# Stress profiles
# =========================================================

def build_stress(profile, n):
    t = np.arange(n)
    s = np.zeros(n, dtype=float)

    if profile == "stable":
        s = 0.08 + 0.03*np.sin(t/4)

    elif profile == "transition":
        s[:10] = 0.10 + 0.02*np.sin(t[:10]/3)
        s[10:] = np.linspace(0.18, 0.52, n-10)

    elif profile == "late_transition":
        s[:20] = 0.09 + 0.02*np.sin(t[:20]/4)
        s[20:] = np.linspace(0.18, 0.50, n-20)

    elif profile == "collapse_recover":
        s[:8] = 0.10 + 0.02*np.sin(t[:8]/3)
        s[8:18] = np.linspace(0.22, 0.88, 10)
        s[18:26] = np.linspace(0.78, 0.32, 8)
        s[26:] = np.linspace(0.28, 0.14, n-26)

    elif profile == "relapse":
        s[:8] = 0.10 + 0.02*np.sin(t[:8]/3)
        s[8:16] = np.linspace(0.20, 0.62, 8)
        s[16:22] = np.linspace(0.55, 0.24, 6)
        s[22:30] = np.linspace(0.28, 0.82, 8)
        s[30:] = np.linspace(0.75, 0.48, n-30)

    else:
        raise ValueError(f"unknown profile: {profile}")

    return np.clip(s, 0, 1)

# =========================================================
# Mapping stress -> signals
# =========================================================

def signals_from_stress(stress, group_heat_boost=0.0):
    n = len(stress)
    noise = lambda scale: np.random.normal(0, scale, n)

    activity   = np.clip(0.86 - 0.48*stress + noise(0.025), 0, 1)
    rumination = np.clip(0.88 - 0.52*stress + noise(0.025), 0, 1)
    locomotion = np.clip(0.91 - 0.50*stress + noise(0.022), 0, 1)
    heat       = np.clip(0.18 + 0.55*stress + group_heat_boost + noise(0.025), 0, 1)

    anomaly = np.clip(0.10 + 0.95*stress + noise(0.03), 0, 1)

    # structural validity should rise when deterioration is coherent,
    # but not be identical to anomaly
    trend_boost = np.r_[0, np.maximum(0, np.diff(stress))]
    validity = np.clip(0.34 + 0.48*stress + 0.25*np.clip(trend_boost*5, 0, 1) + noise(0.02), 0, 1)

    return activity, rumination, locomotion, heat, anomaly, validity

def classify_row(stress_now, stress_prev, anomaly, validity):
    improving = stress_now < stress_prev - 0.04

    if validity < 0.45 or anomaly < 0.25:
        return "GREEN", "stable", "none"

    if anomaly < 0.62:
        if improving:
            return "YELLOW", "recovery-transition", "continue_monitoring"
        else:
            return "YELLOW", "transition", "inspect_next_routine"

    if improving:
        return "YELLOW", "recovery-transition", "continue_monitoring"

    return "RED", "collapse-risk", "immediate_on_site_inspection"

# =========================================================
# Generate animal-level data
# =========================================================

animal_rows = []

group_heat_map = {
    "Pen_B12": 0.03,
    "Pen_C04": 0.04,
    "Pen_D02": 0.00,
}

for group_id, animals in groups.items():
    for animal_id in animals:
        profile = animal_profiles[animal_id]
        stress = build_stress(profile, len(timestamps))
        a, r, l, h, anom, val = signals_from_stress(stress, group_heat_boost=group_heat_map[group_id])

        prev_stress = stress[0]
        for i, ts in enumerate(timestamps):
            color, regime, rec = classify_row(stress[i], prev_stress, anom[i], val[i])
            animal_rows.append({
                "animal_id": animal_id,
                "group_id": group_id,
                "timestamp": ts.isoformat(),
                "welfare_color": color,
                "regime": regime,
                "activity_index": round(float(a[i]), 3),
                "rumination_proxy": round(float(r[i]), 3),
                "locomotion_score": round(float(l[i]), 3),
                "thermal_stress_index": round(float(h[i]), 3),
                "anomaly_score": round(float(anom[i]), 3),
                "validity_score": round(float(val[i]), 3),
                "recommended_action": rec
            })
            prev_stress = stress[i]

animal_df = pd.DataFrame(animal_rows)
animal_out = out_dir / "animal_states_synth_v2.csv"
animal_df.to_csv(animal_out, index=False)

# =========================================================
# Aggregate to group-level
# =========================================================

group_rows = []

for (group_id, ts), g in animal_df.groupby(["group_id", "timestamp"]):
    green = int((g["welfare_color"] == "GREEN").sum())
    yellow = int((g["welfare_color"] == "YELLOW").sum())
    red = int((g["welfare_color"] == "RED").sum())

    group_activity = float(g["activity_index"].mean())
    heat_idx = float(g["thermal_stress_index"].mean())
    competition = float(np.clip(0.15 + 0.08*yellow + 0.12*red + np.random.normal(0, 0.03), 0, 1))
    validity = float(g["validity_score"].mean())
    anomaly = float(g["anomaly_score"].mean())

    if red >= 3 or anomaly >= 0.62:
        welfare_state = "RED"
        regime = "collapse-risk"
        action = "immediate_group_level_intervention"
    elif (yellow + red) >= 3 or anomaly >= 0.34:
        welfare_state = "YELLOW"
        regime = "transition"
        action = "inspect_group_conditions"
    else:
        welfare_state = "GREEN"
        regime = "stable"
        action = "none"

    if welfare_state == "YELLOW" and g["regime"].eq("recovery-transition").mean() > 0.5:
        regime = "recovery-transition"
        action = "continue_monitoring_and_verify_recovery"

    group_rows.append({
        "group_id": group_id,
        "timestamp": ts,
        "welfare_state": welfare_state,
        "regime": regime,
        "animals_green": green,
        "animals_yellow": yellow,
        "animals_red": red,
        "group_activity_index": round(group_activity, 3),
        "competition_proxy": round(competition, 3),
        "heat_stress_index": round(heat_idx, 3),
        "validity_score": round(validity, 3),
        "recommended_action": action
    })

group_df = pd.DataFrame(group_rows).sort_values(["group_id", "timestamp"]).reset_index(drop=True)
group_out = out_dir / "pen_group_states_synth_v2.csv"
group_df.to_csv(group_out, index=False)

# =========================================================
# Aggregate to unit-level
# =========================================================

unit_rows = []

for ts, g in group_df.groupby("timestamp"):
    green = int((g["welfare_state"] == "GREEN").sum())
    yellow = int((g["welfare_state"] == "YELLOW").sum())
    red = int((g["welfare_state"] == "RED").sum())

    thi = int(round(64 + 8*g["heat_stress_index"].mean() * 10 / 10))
    valid = float(g["validity_score"].mean())

    if red >= 2:
        welfare_state = "RED"
        regime = "collapse-risk"
        global_alert = "high"
        ventilation = "inadequate"
        water = "compromised"
        feeding = "stressed"
        bedding = "degraded"
        action = "immediate_unit_level_intervention"
    elif (yellow + red) >= 2:
        welfare_state = "YELLOW"
        regime = "transition"
        global_alert = "moderate"
        ventilation = "stressed"
        water = "under_pressure"
        feeding = "normal"
        bedding = "under_review"
        action = "review_climate_settings_and_monitor_groups"
    else:
        welfare_state = "GREEN"
        regime = "stable"
        global_alert = "low"
        ventilation = "normal"
        water = "normal"
        feeding = "normal"
        bedding = "acceptable"
        action = "none"

    if welfare_state == "YELLOW" and g["regime"].eq("recovery-transition").mean() > 0.5:
        regime = "recovery-transition"
        action = "continue_monitoring_and_verify_recovery"

    highest = g.sort_values(["animals_red", "animals_yellow"], ascending=False)["group_id"].tolist()
    highest_risk_groups = ";".join(highest[:3]) if welfare_state != "GREEN" else "none"

    unit_rows.append({
        "unit_id": "Farm_Unit_SYNTH_1",
        "timestamp": ts,
        "welfare_state": welfare_state,
        "regime": regime,
        "global_alert_level": global_alert,
        "THI": thi,
        "ventilation_state": ventilation,
        "water_system_status": water,
        "feeding_system_status": feeding,
        "bedding_status": bedding,
        "highest_risk_groups": highest_risk_groups,
        "validity_score": round(valid, 3),
        "recommended_action": action
    })

unit_df = pd.DataFrame(unit_rows).sort_values("timestamp").reset_index(drop=True)
unit_out = out_dir / "unit_farm_states_synth_v2.csv"
unit_df.to_csv(unit_out, index=False)

print("\n=== SYNTH DATASET V2 GENERATED ===")
print(f"animal rows : {len(animal_df)}")
print(f"group rows  : {len(group_df)}")
print(f"unit rows   : {len(unit_df)}")

print("\nanimal welfare distribution:")
print(animal_df["welfare_color"].value_counts())

print("\ngroup welfare distribution:")
print(group_df["welfare_state"].value_counts())

print("\nunit welfare distribution:")
print(unit_df["welfare_state"].value_counts())

print(f"\nSaved:")
print(f"- {animal_out}")
print(f"- {group_out}")
print(f"- {unit_out}")

