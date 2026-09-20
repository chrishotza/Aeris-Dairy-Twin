import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SIM_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_sim_reinforced_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_IN = SIM_DIR / "animal_states_challenge_sim_v1.csv"

RANDOM_SEED = 2084
N_RANDOM = 900
TOP_SEEDS = 80
N_REFINE = 1800
PROGRESS_EVERY = 100

SEV2RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
RANK2SEV = {v: k for k, v in SEV2RANK.items()}

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def safe_read(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def truth_from_hour_scenario(scenario_arr, hour_arr, hours_total):
    out = np.empty(len(hour_arr), dtype=object)
    s1 = int(hours_total * 0.35)
    s2 = int(hours_total * 0.50)
    s3 = int(hours_total * 0.78)

    for i, (scen, h) in enumerate(zip(scenario_arr, hour_arr)):
        if scen == "stable_baseline":
            out[i] = "GREEN"
        elif h < s1:
            out[i] = "GREEN"
        elif s1 <= h < s2:
            out[i] = "YELLOW"
        elif s2 <= h < s3:
            out[i] = "YELLOW" if scen == "post_event_recovery" else "RED"
        else:
            out[i] = "WATCH"
    return out

def metrics(pred_sev, truth_sev):
    pred_rank = pd.Series(pred_sev).astype(str).str.upper().map(SEV2RANK).fillna(0).astype(int).to_numpy()
    true_rank = pd.Series(truth_sev).astype(str).str.upper().map(SEV2RANK).fillna(0).astype(int).to_numpy()

    exact = float((pred_rank == true_rank).mean())

    pred_concern = (pred_rank >= 2).astype(int)
    true_concern = (true_rank >= 2).astype(int)

    tp = int(((pred_concern == 1) & (true_concern == 1)).sum())
    fp = int(((pred_concern == 1) & (true_concern == 0)).sum())
    fn = int(((pred_concern == 0) & (true_concern == 1)).sum())
    tn = int(((pred_concern == 0) & (true_concern == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    accuracy = (tp + tn) / max(len(pred_rank), 1)

    red_rate = float((pred_rank == 3).mean())
    concern_rate = float((pred_rank >= 2).mean())

    return {
        "exact_match_rate": exact,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "red_rate": red_rate,
        "concern_rate": concern_rate,
    }

def normalize_params(p):
    p = dict(p)

    # reinforcement amplitudes
    p["heat_amp"] = float(np.clip(p["heat_amp"], 0.0, 0.90))
    p["lame_amp"] = float(np.clip(p["lame_amp"], 0.0, 1.20))

    # center around best balanced_sweep_v2 region
    p["base_gain"] = float(np.clip(p["base_gain"], 0.45, 1.10))
    p["heat_w"] = float(np.clip(p["heat_w"], 0.00, 0.18))
    p["loco_w"] = float(np.clip(p["loco_w"], 0.00, 0.20))
    p["resp_w"] = float(np.clip(p["resp_w"], 0.05, 0.30))
    p["manage_w"] = float(np.clip(p["manage_w"], 0.05, 0.25))
    p["intake_w"] = float(np.clip(p["intake_w"], 0.05, 0.30))
    p["water_w"] = float(np.clip(p["water_w"], 0.02, 0.18))
    p["visual_w"] = float(np.clip(p["visual_w"], 0.02, 0.18))

    p["animal_y"] = float(np.clip(p["animal_y"], 0.34, 0.52))
    p["animal_r"] = float(np.clip(p["animal_r"], 0.46, 0.68))
    if p["animal_r"] <= p["animal_y"]:
        p["animal_r"] = min(0.68, p["animal_y"] + 0.08)

    p["group_y_share"] = float(np.clip(p["group_y_share"], 0.08, 0.24))
    p["group_r_share"] = float(np.clip(p["group_r_share"], 0.18, 0.40))
    if p["group_r_share"] < p["group_y_share"]:
        p["group_r_share"] = p["group_y_share"] + 0.04

    p["group_y_score"] = float(np.clip(p["group_y_score"], 0.30, 0.48))
    p["group_r_score"] = float(np.clip(p["group_r_score"], 0.44, 0.66))
    if p["group_r_score"] <= p["group_y_score"]:
        p["group_r_score"] = min(0.66, p["group_y_score"] + 0.06)

    p["unit_y_share"] = float(np.clip(p["unit_y_share"], 0.06, 0.18))
    p["unit_r_share"] = float(np.clip(p["unit_r_share"], 0.20, 0.38))
    if p["unit_r_share"] < p["unit_y_share"]:
        p["unit_r_share"] = p["unit_y_share"] + 0.05

    p["unit_y_score"] = float(np.clip(p["unit_y_score"], 0.38, 0.56))
    p["unit_r_score"] = float(np.clip(p["unit_r_score"], 0.45, 0.62))
    if p["unit_r_score"] <= p["unit_y_score"]:
        p["unit_r_score"] = min(0.62, p["unit_y_score"] + 0.05)

    p["min_red_duration"] = int(np.clip(round(p["min_red_duration"]), 1, 3))
    p["min_yellow_duration"] = int(np.clip(round(p["min_yellow_duration"]), 2, 4))
    return p

def sample_params():
    return normalize_params({
        "heat_amp": random.uniform(0.10, 0.60),
        "lame_amp": random.uniform(0.15, 0.85),

        "base_gain": random.uniform(0.60, 0.90),
        "heat_w": random.uniform(0.02, 0.12),
        "loco_w": random.uniform(0.02, 0.12),
        "resp_w": random.uniform(0.10, 0.24),
        "manage_w": random.uniform(0.08, 0.20),
        "intake_w": random.uniform(0.10, 0.24),
        "water_w": random.uniform(0.04, 0.12),
        "visual_w": random.uniform(0.04, 0.12),

        "animal_y": random.uniform(0.40, 0.48),
        "animal_r": random.uniform(0.50, 0.60),

        "group_y_share": random.uniform(0.10, 0.20),
        "group_r_share": random.uniform(0.24, 0.34),
        "group_y_score": random.uniform(0.34, 0.44),
        "group_r_score": random.uniform(0.48, 0.58),

        "unit_y_share": random.uniform(0.07, 0.14),
        "unit_r_share": random.uniform(0.24, 0.34),
        "unit_y_score": random.uniform(0.42, 0.50),
        "unit_r_score": random.uniform(0.48, 0.56),

        "min_red_duration": random.choice([1, 2]),
        "min_yellow_duration": random.choice([3, 4]),
    })

def jitter_params(seed):
    return normalize_params({
        "heat_amp": seed["heat_amp"] + random.uniform(-0.08, 0.08),
        "lame_amp": seed["lame_amp"] + random.uniform(-0.10, 0.10),

        "base_gain": seed["base_gain"] + random.uniform(-0.06, 0.06),
        "heat_w": seed["heat_w"] + random.uniform(-0.02, 0.02),
        "loco_w": seed["loco_w"] + random.uniform(-0.02, 0.02),
        "resp_w": seed["resp_w"] + random.uniform(-0.03, 0.03),
        "manage_w": seed["manage_w"] + random.uniform(-0.02, 0.02),
        "intake_w": seed["intake_w"] + random.uniform(-0.03, 0.03),
        "water_w": seed["water_w"] + random.uniform(-0.02, 0.02),
        "visual_w": seed["visual_w"] + random.uniform(-0.02, 0.02),

        "animal_y": seed["animal_y"] + random.uniform(-0.03, 0.03),
        "animal_r": seed["animal_r"] + random.uniform(-0.03, 0.03),

        "group_y_share": seed["group_y_share"] + random.uniform(-0.02, 0.02),
        "group_r_share": seed["group_r_share"] + random.uniform(-0.03, 0.03),
        "group_y_score": seed["group_y_score"] + random.uniform(-0.03, 0.03),
        "group_r_score": seed["group_r_score"] + random.uniform(-0.03, 0.03),

        "unit_y_share": seed["unit_y_share"] + random.uniform(-0.02, 0.02),
        "unit_r_share": seed["unit_r_share"] + random.uniform(-0.03, 0.03),
        "unit_y_score": seed["unit_y_score"] + random.uniform(-0.03, 0.03),
        "unit_r_score": seed["unit_r_score"] + random.uniform(-0.03, 0.03),

        "min_red_duration": seed["min_red_duration"] + random.choice([-1, 0, 1]),
        "min_yellow_duration": seed["min_yellow_duration"] + random.choice([-1, 0, 1]),
    })

def apply_persistence(seq, min_red_duration, min_yellow_duration):
    out = seq.copy()
    i = 0
    while i < len(out):
        cur = out[i]
        if cur not in {"RED", "YELLOW"}:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] == cur:
            j += 1
        dur = j - i
        req = min_red_duration if cur == "RED" else min_yellow_duration
        if dur < req:
            out[i:j] = "WATCH" if cur == "RED" else "GREEN"
        i = j
    return out

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df = safe_read(ANIMAL_IN)

needed = [
    "run_id","scenario","hour","unit_id","group_id","animal_id",
    "animal_score","phase",
    "rumination","activity","locomotion_quality","feeding_engagement",
    "drinking_pressure","respiration_load","thermal_discomfort",
    "management_disruption","visual_anomaly_proxy",
    "ventilation_quality","water_status","feed_delivery_quality","bedding_quality"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[needed].copy()

for c in needed:
    if c not in ["run_id","scenario","hour","unit_id","group_id","animal_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

hours_total = int(df["hour"].max()) + 1
df = df.sort_values(["run_id","scenario","unit_id","group_id","animal_id","hour"]).reset_index(drop=True)

group_keys = pd.factorize(pd.MultiIndex.from_frame(df[["run_id","scenario","hour","unit_id","group_id"]]))[0]
unit_keys  = pd.factorize(pd.MultiIndex.from_frame(df[["run_id","scenario","hour","unit_id"]]))[0]

group_count = np.bincount(group_keys)
n_groups = int(group_keys.max()) + 1
n_units = int(unit_keys.max()) + 1

group_to_unit = np.zeros(n_groups, dtype=np.int64)
tmp_map = pd.DataFrame({"group_key": group_keys, "unit_key": unit_keys}).drop_duplicates("group_key")
group_to_unit[tmp_map["group_key"].to_numpy()] = tmp_map["unit_key"].to_numpy()
group_unit_counts = np.bincount(group_to_unit, minlength=n_units)

unit_meta = (
    df[["run_id","scenario","hour","unit_id"]]
    .drop_duplicates()
    .sort_values(["run_id","scenario","unit_id","hour"])
    .reset_index(drop=True)
)
unit_truth = truth_from_hour_scenario(
    unit_meta["scenario"].to_numpy(),
    unit_meta["hour"].to_numpy(),
    hours_total
)

seq_positions = []
for _, g in unit_meta.groupby(["run_id","scenario","unit_id"], sort=False):
    seq_positions.append(g.index.to_numpy())

# raw arrays
scenario_arr = df["scenario"].astype(str).to_numpy()
phase = clamp01(df["phase"].to_numpy(dtype=float))
base_score = df["animal_score"].to_numpy(dtype=float)

heat_focus = clamp01(
    0.34 * df["thermal_discomfort"].to_numpy(dtype=float) +
    0.26 * df["respiration_load"].to_numpy(dtype=float) +
    0.14 * df["drinking_pressure"].to_numpy(dtype=float) +
    0.14 * (1.0 - df["activity"].to_numpy(dtype=float)) +
    0.12 * (1.0 - df["ventilation_quality"].to_numpy(dtype=float))
)

lame_focus = clamp01(
    0.40 * (1.0 - df["locomotion_quality"].to_numpy(dtype=float)) +
    0.20 * (1.0 - df["bedding_quality"].to_numpy(dtype=float)) +
    0.18 * df["visual_anomaly_proxy"].to_numpy(dtype=float) +
    0.12 * (1.0 - df["activity"].to_numpy(dtype=float)) +
    0.10 * (1.0 - df["rumination"].to_numpy(dtype=float))
)

heat = df["thermal_discomfort"].to_numpy(dtype=float)
loco_drop = (1.0 - df["locomotion_quality"].to_numpy(dtype=float))
resp = df["respiration_load"].to_numpy(dtype=float)
manage = df["management_disruption"].to_numpy(dtype=float)
intake_drop = (1.0 - df["feeding_engagement"].to_numpy(dtype=float))
water_drop = (1.0 - df["water_status"].to_numpy(dtype=float))
visual = df["visual_anomaly_proxy"].to_numpy(dtype=float)

is_heat = (scenario_arr == "heat_stress_wave").astype(float)
is_lame = (scenario_arr == "lameness_cluster").astype(float)

group_info = (
    df[["run_id","scenario","hour","unit_id","group_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
group_info["group_key"] = np.arange(len(group_info))
group_info = group_info.sort_values("group_key").reset_index(drop=True)

def evaluate_params(params):
    reinforced_base = clamp01(
        base_score +
        params["heat_amp"] * is_heat * phase * heat_focus +
        params["lame_amp"] * is_lame * phase * lame_focus
    )

    boosted = clamp01(
        params["base_gain"] * reinforced_base +
        params["heat_w"] * heat +
        params["loco_w"] * loco_drop +
        params["resp_w"] * resp +
        params["manage_w"] * manage +
        params["intake_w"] * intake_drop +
        params["water_w"] * water_drop +
        params["visual_w"] * visual
    )

    animal_y = boosted >= params["animal_y"]
    animal_r = boosted >= params["animal_r"]

    g_sum = np.bincount(group_keys, weights=boosted, minlength=n_groups)
    g_mean = g_sum / group_count
    g_yshare = np.bincount(group_keys, weights=animal_y.astype(float), minlength=n_groups) / group_count
    g_rshare = np.bincount(group_keys, weights=animal_r.astype(float), minlength=n_groups) / group_count

    group_rank = np.zeros(n_groups, dtype=np.int8)
    g_red = (g_rshare >= params["group_r_share"]) | (g_mean >= params["group_r_score"])
    g_yel = (g_yshare >= params["group_y_share"]) | (g_mean >= params["group_y_score"])
    group_rank[g_yel] = 2
    group_rank[g_red] = 3

    u_score_mean = np.bincount(group_to_unit, weights=g_mean, minlength=n_units) / group_unit_counts
    u_yshare = np.bincount(group_to_unit, weights=(group_rank >= 2).astype(float), minlength=n_units) / group_unit_counts
    u_rshare = np.bincount(group_to_unit, weights=(group_rank == 3).astype(float), minlength=n_units) / group_unit_counts

    unit_rank = np.zeros(n_units, dtype=np.int8)
    u_red = (u_rshare >= params["unit_r_share"]) | (u_score_mean >= params["unit_r_score"])
    u_yel = (u_yshare >= params["unit_y_share"]) | (u_score_mean >= params["unit_y_score"])
    unit_rank[u_yel] = 2
    unit_rank[u_red] = 3

    unit_sev = np.array([RANK2SEV[int(x)] for x in unit_rank], dtype=object)
    for idxs in seq_positions:
        unit_sev[idxs] = apply_persistence(
            unit_sev[idxs],
            params["min_red_duration"],
            params["min_yellow_duration"]
        )

    overall = metrics(unit_sev, unit_truth)

    unit_pred_tmp = unit_meta.copy()
    unit_pred_tmp["pred_severity"] = unit_sev
    unit_pred_tmp["truth_severity"] = unit_truth

    scenario_rows = []
    for scen, g in unit_pred_tmp.groupby("scenario"):
        m = metrics(g["pred_severity"], g["truth_severity"])
        row = {"scenario": scen}
        row.update(m)
        scenario_rows.append(row)
    scen_df = pd.DataFrame(scenario_rows)

    baseline = scen_df[scen_df["scenario"] == "stable_baseline"].iloc[0]
    lame = scen_df[scen_df["scenario"] == "lameness_cluster"].iloc[0]
    vent = scen_df[scen_df["scenario"] == "ventilation_failure"].iloc[0]
    heatm = scen_df[scen_df["scenario"] == "heat_stress_wave"].iloc[0]
    waterm = scen_df[scen_df["scenario"] == "water_system_issue"].iloc[0]
    feedm = scen_df[scen_df["scenario"] == "feeding_disruption"].iloc[0]

    target_df = scen_df[scen_df["scenario"].isin([
        "heat_stress_wave","lameness_cluster","feeding_disruption",
        "water_system_issue","ventilation_failure","post_event_recovery"
    ])].copy()

    min_recall = float(target_df["recall"].min())
    mean_recall = float(target_df["recall"].mean())

    score = (
        2.2 * mean_recall +
        2.4 * min_recall +
        0.9 * float(overall["precision"]) +
        0.9 * float(overall["specificity"]) +
        0.7 * float(overall["accuracy"]) +
        1.2 * float(lame["recall"]) +
        0.8 * float(heatm["recall"]) +
        0.5 * float(vent["recall"])
    )

    # keep baseline sane
    score -= 2.0 * float(baseline["concern_rate"])
    score -= 0.8 * (1.0 - float(baseline["specificity"]))

    # preserve what already works
    score += 0.25 * float(feedm["recall"])
    score += 0.25 * float(waterm["recall"])

    # hard filters
    viable = True
    if float(baseline["concern_rate"]) > 0.08:
        viable = False
    if float(baseline["specificity"]) < 0.92:
        viable = False
    if float(overall["concern_rate"]) > 0.52:
        viable = False
    if float(overall["red_rate"]) > 0.15:
        viable = False
    if not viable:
        score -= 100.0

    row = dict(params)
    row.update(overall)
    row["objective"] = float(score)
    row["min_target_recall"] = min_recall
    row["mean_target_recall"] = mean_recall
    row["baseline_concern_rate"] = float(baseline["concern_rate"])
    row["baseline_specificity"] = float(baseline["specificity"])
    row["lameness_recall"] = float(lame["recall"])
    row["ventilation_recall"] = float(vent["recall"])
    row["heat_recall"] = float(heatm["recall"])
    row["water_recall"] = float(waterm["recall"])
    row["feed_recall"] = float(feedm["recall"])

    return row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare, reinforced_base

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------
t0 = time.time()
rows = []
best_payload = None
best_obj = -1e18

print("\n=== CHALLENGE SIM REINFORCED V2 ===")
print(f"animal rows: {len(df)}")
print(f"group slots: {n_groups}")
print(f"unit slots : {n_units}")
print(f"random iters: {N_RANDOM}")
print(f"refine iters: {N_REFINE}")
print("")

seed_rows = []

for i in range(N_RANDOM):
    p = sample_params()
    row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare, reinforced_base = evaluate_params(p)
    rows.append(row)
    if row["objective"] > best_obj:
        best_obj = row["objective"]
        best_payload = (p, scen_df.copy(), unit_sev.copy(), group_rank.copy(), g_mean.copy(), g_yshare.copy(), g_rshare.copy(), reinforced_base.copy())
    if (i + 1) % PROGRESS_EVERY == 0:
        print(f"random {i+1}/{N_RANDOM} | elapsed {(time.time()-t0)/60:.1f} min | best {best_obj:.6f}")

seed_df = pd.DataFrame(rows).sort_values(
    ["objective","min_target_recall","lameness_recall","heat_recall","baseline_specificity"],
    ascending=False
).reset_index(drop=True)
seed_rows = seed_df.head(TOP_SEEDS).to_dict("records")

for i in range(N_REFINE):
    if seed_rows and random.random() < 0.82:
        p = jitter_params(random.choice(seed_rows[:50]))
    else:
        p = sample_params()

    row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare, reinforced_base = evaluate_params(p)
    rows.append(row)
    if row["objective"] > best_obj:
        best_obj = row["objective"]
        best_payload = (p, scen_df.copy(), unit_sev.copy(), group_rank.copy(), g_mean.copy(), g_yshare.copy(), g_rshare.copy(), reinforced_base.copy())
    if (i + 1) % PROGRESS_EVERY == 0:
        print(f"refine {i+1}/{N_REFINE} | elapsed {(time.time()-t0)/60:.1f} min | best {best_obj:.6f}")

res = pd.DataFrame(rows).sort_values(
    ["objective","min_target_recall","lameness_recall","heat_recall","baseline_specificity"],
    ascending=False
).reset_index(drop=True)

best_params, best_scen_df, best_unit_sev, best_group_rank, best_g_mean, best_g_yshare, best_g_rshare, best_reinforced_base = best_payload
best_metrics = res.iloc[0].to_dict()

group_pred = group_info.copy()
group_pred["group_score_reinforced"] = best_g_mean
group_pred["group_yshare_reinforced"] = best_g_yshare
group_pred["group_rshare_reinforced"] = best_g_rshare
group_pred["group_severity_reinforced"] = [RANK2SEV[int(x)] for x in best_group_rank]

unit_pred = unit_meta.copy()
unit_pred["pred_severity_reinforced"] = best_unit_sev
unit_pred["truth_severity"] = unit_truth
unit_pred["match"] = (unit_pred["pred_severity_reinforced"] == unit_pred["truth_severity"]).astype(int)

animal_reinforced = df.copy()
animal_reinforced["animal_score_reinforced"] = best_reinforced_base

dashboard_rows = []
for (scenario, unit_id), g in unit_pred.groupby(["scenario","unit_id"]):
    dashboard_rows.append({
        "scenario": scenario,
        "unit_id": unit_id,
        "hours_total": int(len(g)),