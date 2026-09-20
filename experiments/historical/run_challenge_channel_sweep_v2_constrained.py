import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SIM_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_channel_sweep_v2_constrained"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_IN = SIM_DIR / "animal_states_challenge_sim_v1.csv"

RANDOM_SEED = 2073
N_RANDOM = 1000
TOP_SEEDS = 80
N_REFINE = 2200
PROGRESS_EVERY = 100

SEV2RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
RANK2SEV = {v: k for k, v in SEV2RANK.items()}

TARGET_SCENARIOS = [
    "heat_stress_wave",
    "lameness_cluster",
    "feeding_disruption",
    "water_system_issue",
    "ventilation_failure",
    "post_event_recovery",
]

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

    for k in [
        "heat_gain","lame_gain","feed_gain","water_gain","vent_gain","manage_gain",
        "base_gain","visual_gain"
    ]:
        p[k] = float(np.clip(p[k], 0.0, 2.5))

    p["animal_y"] = float(np.clip(p["animal_y"], 0.20, 0.65))
    p["animal_r"] = float(np.clip(p["animal_r"], 0.30, 0.90))
    if p["animal_r"] <= p["animal_y"]:
        p["animal_r"] = min(0.90, p["animal_y"] + 0.08)

    p["group_y_share"] = float(np.clip(p["group_y_share"], 0.05, 0.40))
    p["group_r_share"] = float(np.clip(p["group_r_share"], 0.08, 0.50))
    if p["group_r_share"] < p["group_y_share"]:
        p["group_r_share"] = p["group_y_share"] + 0.03

    p["group_y_score"] = float(np.clip(p["group_y_score"], 0.20, 0.75))
    p["group_r_score"] = float(np.clip(p["group_r_score"], 0.30, 0.95))
    if p["group_r_score"] <= p["group_y_score"]:
        p["group_r_score"] = min(0.95, p["group_y_score"] + 0.06)

    p["unit_y_share"] = float(np.clip(p["unit_y_share"], 0.05, 0.50))
    p["unit_r_share"] = float(np.clip(p["unit_r_share"], 0.08, 0.60))
    if p["unit_r_share"] < p["unit_y_share"]:
        p["unit_r_share"] = p["unit_y_share"] + 0.03

    p["unit_y_score"] = float(np.clip(p["unit_y_score"], 0.20, 0.85))
    p["unit_r_score"] = float(np.clip(p["unit_r_score"], 0.30, 0.98))
    if p["unit_r_score"] <= p["unit_y_score"]:
        p["unit_r_score"] = min(0.98, p["unit_y_score"] + 0.06)

    p["min_red_duration"] = int(np.clip(round(p["min_red_duration"]), 1, 4))
    p["min_yellow_duration"] = int(np.clip(round(p["min_yellow_duration"]), 1, 4))
    return p

def sample_params():
    return normalize_params({
        "heat_gain": random.uniform(0.6, 1.8),
        "lame_gain": random.uniform(0.8, 2.2),
        "feed_gain": random.uniform(0.6, 1.8),
        "water_gain": random.uniform(0.6, 1.8),
        "vent_gain": random.uniform(0.8, 2.2),
        "manage_gain": random.uniform(0.4, 1.6),
        "base_gain": random.uniform(0.2, 1.0),
        "visual_gain": random.uniform(0.2, 1.2),

        "animal_y": random.uniform(0.24, 0.50),
        "animal_r": random.uniform(0.38, 0.74),

        "group_y_share": random.uniform(0.06, 0.24),
        "group_r_share": random.uniform(0.10, 0.34),
        "group_y_score": random.uniform(0.26, 0.56),
        "group_r_score": random.uniform(0.36, 0.72),

        "unit_y_share": random.uniform(0.06, 0.24),
        "unit_r_share": random.uniform(0.10, 0.34),
        "unit_y_score": random.uniform(0.26, 0.56),
        "unit_r_score": random.uniform(0.36, 0.72),

        "min_red_duration": random.choice([1, 2, 3]),
        "min_yellow_duration": random.choice([2, 3, 4]),
    })

def jitter_params(seed):
    return normalize_params({
        "heat_gain": seed["heat_gain"] + random.uniform(-0.20, 0.20),
        "lame_gain": seed["lame_gain"] + random.uniform(-0.25, 0.25),
        "feed_gain": seed["feed_gain"] + random.uniform(-0.20, 0.20),
        "water_gain": seed["water_gain"] + random.uniform(-0.20, 0.20),
        "vent_gain": seed["vent_gain"] + random.uniform(-0.25, 0.25),
        "manage_gain": seed["manage_gain"] + random.uniform(-0.15, 0.15),
        "base_gain": seed["base_gain"] + random.uniform(-0.10, 0.10),
        "visual_gain": seed["visual_gain"] + random.uniform(-0.12, 0.12),

        "animal_y": seed["animal_y"] + random.uniform(-0.04, 0.04),
        "animal_r": seed["animal_r"] + random.uniform(-0.05, 0.05),

        "group_y_share": seed["group_y_share"] + random.uniform(-0.03, 0.03),
        "group_r_share": seed["group_r_share"] + random.uniform(-0.03, 0.03),
        "group_y_score": seed["group_y_score"] + random.uniform(-0.04, 0.04),
        "group_r_score": seed["group_r_score"] + random.uniform(-0.04, 0.04),

        "unit_y_share": seed["unit_y_share"] + random.uniform(-0.03, 0.03),
        "unit_r_share": seed["unit_r_share"] + random.uniform(-0.03, 0.03),
        "unit_y_score": seed["unit_y_score"] + random.uniform(-0.04, 0.04),
        "unit_r_score": seed["unit_r_score"] + random.uniform(-0.04, 0.04),

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
    "run_id","scenario","hour","unit_id","group_id","animal_id","animal_score",
    "rumination","activity","locomotion_quality","feeding_engagement",
    "drinking_pressure","respiration_load","thermal_discomfort",
    "management_disruption","visual_anomaly_proxy",
    "ventilation_quality","water_status","feed_delivery_quality","bedding_quality"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in animal simulation file: {missing}")

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

# raw channels
base_score = df["animal_score"].to_numpy(dtype=float)

heat_head = clamp01(
    0.42 * df["thermal_discomfort"].to_numpy(dtype=float) +
    0.28 * df["respiration_load"].to_numpy(dtype=float) +
    0.12 * df["drinking_pressure"].to_numpy(dtype=float) +
    0.10 * (1.0 - df["activity"].to_numpy(dtype=float)) +
    0.08 * (1.0 - df["ventilation_quality"].to_numpy(dtype=float))
)

lame_head = clamp01(
    0.48 * (1.0 - df["locomotion_quality"].to_numpy(dtype=float)) +
    0.18 * (1.0 - df["bedding_quality"].to_numpy(dtype=float)) +
    0.16 * df["visual_anomaly_proxy"].to_numpy(dtype=float) +
    0.10 * (1.0 - df["activity"].to_numpy(dtype=float)) +
    0.08 * (1.0 - df["rumination"].to_numpy(dtype=float))
)

feed_head = clamp01(
    0.34 * (1.0 - df["feeding_engagement"].to_numpy(dtype=float)) +
    0.28 * (1.0 - df["feed_delivery_quality"].to_numpy(dtype=float)) +
    0.20 * (1.0 - df["rumination"].to_numpy(dtype=float)) +
    0.10 * df["management_disruption"].to_numpy(dtype=float) +
    0.08 * (1.0 - df["activity"].to_numpy(dtype=float))
)

water_head = clamp01(
    0.36 * df["drinking_pressure"].to_numpy(dtype=float) +
    0.28 * (1.0 - df["water_status"].to_numpy(dtype=float)) +
    0.18 * df["thermal_discomfort"].to_numpy(dtype=float) +
    0.10 * df["respiration_load"].to_numpy(dtype=float) +
    0.08 * (1.0 - df["activity"].to_numpy(dtype=float))
)

vent_head = clamp01(
    0.34 * (1.0 - df["ventilation_quality"].to_numpy(dtype=float)) +
    0.30 * df["respiration_load"].to_numpy(dtype=float) +
    0.20 * df["thermal_discomfort"].to_numpy(dtype=float) +
    0.08 * df["visual_anomaly_proxy"].to_numpy(dtype=float) +
    0.08 * (1.0 - df["activity"].to_numpy(dtype=float))
)

manage_head = clamp01(
    0.46 * df["management_disruption"].to_numpy(dtype=float) +
    0.18 * (1.0 - df["feed_delivery_quality"].to_numpy(dtype=float)) +
    0.18 * (1.0 - df["water_status"].to_numpy(dtype=float)) +
    0.10 * (1.0 - df["bedding_quality"].to_numpy(dtype=float)) +
    0.08 * (1.0 - df["activity"].to_numpy(dtype=float))
)

visual_head = df["visual_anomaly_proxy"].to_numpy(dtype=float)

group_info = (
    df[["run_id","scenario","hour","unit_id","group_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
group_info["group_key"] = np.arange(len(group_info))
group_info = group_info.sort_values("group_key").reset_index(drop=True)

# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------
def evaluate_params(params):
    composite = clamp01(
        params["base_gain"] * base_score +
        params["heat_gain"] * heat_head +
        params["lame_gain"] * lame_head +
        params["feed_gain"] * feed_head +
        params["water_gain"] * water_head +
        params["vent_gain"] * vent_head +
        params["manage_gain"] * manage_head +
        params["visual_gain"] * visual_head
    )

    animal_y = composite >= params["animal_y"]
    animal_r = composite >= params["animal_r"]

    g_sum = np.bincount(group_keys, weights=composite, minlength=n_groups)
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
    heat = scen_df[scen_df["scenario"] == "heat_stress_wave"].iloc[0]

    target_df = scen_df[scen_df["scenario"].isin(TARGET_SCENARIOS)].copy()
    min_recall = float(target_df["recall"].min())
    mean_recall = float(target_df["recall"].mean())
    mean_precision = float(target_df["precision"].mean())
    mean_specificity = float(target_df["specificity"].mean())

    # hard viability gates first
    baseline_concern = float(baseline["concern_rate"])
    baseline_spec = float(baseline["specificity"])
    overall_concern = float(overall["concern_rate"])
    overall_red = float(overall["red_rate"])
    overall_spec = float(overall["specificity"])

    viable = True
    if baseline_concern > 0.08:
        viable = False
    if baseline_spec < 0.92:
        viable = False
    if overall_concern > 0.50:
        viable = False
    if overall_red > 0.12:
        viable = False
    if overall_spec < 0.72:
        viable = False

    score = (
        1.8 * mean_recall +
        2.0 * min_recall +
        1.1 * mean_precision +
        1.0 * mean_specificity +
        0.7 * overall["accuracy"] +
        1.0 * float(lame["recall"]) +
        1.0 * float(vent["recall"]) +
        0.6 * float(heat["recall"])
    )

    score -= 1.6 * baseline_concern
    score -= 0.8 * (1.0 - baseline_spec)

    if overall_concern > 0.42:
        score -= 2.0 * (overall_concern - 0.42)
    if overall_red > 0.08:
        score -= 2.5 * (overall_red - 0.08)

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
    row["heat_recall"] = float(heat["recall"])

    return row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------
t0 = time.time()
rows = []
best_payload = None
best_obj = -1e18

print("\n=== CHALLENGE CHANNEL SWEEP V2 CONSTRAINED ===")
print(f"animal rows: {len(df)}")
print(f"group slots: {n_groups}")
print(f"unit slots : {n_units}")
print(f"random iters: {N_RANDOM}")
print(f"refine iters: {N_REFINE}")
print("")

seed_rows = []

for i in range(N_RANDOM):
    p = sample_params()
    row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare = evaluate_params(p)
    rows.append(row)
    if row["objective"] > best_obj:
        best_obj = row["objective"]
        best_payload = (p, scen_df.copy(), unit_sev.copy(), group_rank.copy(), g_mean.copy(), g_yshare.copy(), g_rshare.copy())
    if (i + 1) % PROGRESS_EVERY == 0:
        print(f"random {i+1}/{N_RANDOM} | elapsed {(time.time()-t0)/60:.1f} min | best {best_obj:.6f}")

seed_df = pd.DataFrame(rows).sort_values(
    ["objective","min_target_recall","lameness_recall","ventilation_recall","baseline_specificity"],
    ascending=False
).reset_index(drop=True)
seed_rows = seed_df.head(TOP_SEEDS).to_dict("records")

for i in range(N_REFINE):
    if seed_rows and random.random() < 0.82:
        p = jitter_params(random.choice(seed_rows[:55]))
    else:
        p = sample_params()

    row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare = evaluate_params(p)
    rows.append(row)
    if row["objective"] > best_obj:
        best_obj = row["objective"]
        best_payload = (p, scen_df.copy(), unit_sev.copy(), group_rank.copy(), g_mean.copy(), g_yshare.copy(), g_rshare.copy())
    if (i + 1) % PROGRESS_EVERY == 0:
        print(f"refine {i+1}/{N_REFINE} | elapsed {(time.time()-t0)/60:.1f} min | best {best_obj:.6f}")

res = pd.DataFrame(rows).sort_values(
    ["objective","min_target_recall","lameness_recall","ventilation_recall","baseline_specificity"],
    ascending=False
).reset_index(drop=True)

best_params, best_scen_df, best_unit_sev, best_group_rank, best_g_mean, best_g_yshare, best_g_rshare = best_payload
best_metrics = res.iloc[0].to_dict()

group_pred = group_info.copy()
group_pred["group_score_channel"] = best_g_mean
group_pred["group_yshare_channel"] = best_g_yshare
group_pred["group_rshare_channel"] = best_g_rshare
group_pred["group_severity_channel"] = [RANK2SEV[int(x)] for x in best_group_rank]

unit_pred = unit_meta.copy()
unit_pred["pred_severity_channel"] = best_unit_sev
unit_pred["truth_severity"] = unit_truth
unit_pred["match"] = (unit_pred["pred_severity_channel"] == unit_pred["truth_severity"]).astype(int)

dashboard_rows = []
for (scenario, unit_id), g in unit_pred.groupby(["scenario","unit_id"]):
    dashboard_rows.append({
        "scenario": scenario,
        "unit_id": unit_id,
        "hours_total": int(len(g)),
        "hours_green": int((g["pred_severity_channel"] == "GREEN").sum()),
        "hours_yellow": int((g["pred_severity_channel"] == "YELLOW").sum()),
        "hours_red": int((g["pred_severity_channel"] == "RED").sum()),
        "hours_concern": int(g["pred_severity_channel"].isin(["YELLOW","RED"]).sum()),
        "concern_rate": round(float(g["pred_severity_channel"].isin(["YELLOW","RED"]).mean()), 4),
        "red_rate": round(float((g["pred_severity_channel"] == "RED").mean()), 4),
        "match_rate": round(float(g["match"].mean()), 4),
    })
dashboard_df = pd.DataFrame(dashboard_rows).sort_values(["scenario","unit_id"]).reset_index(drop=True)

leaderboard_out = OUT_DIR / "challenge_channel_sweep_leaderboard_v1.csv"
group_out = OUT_DIR / "challenge_channel_group_predictions_v2.csv"
unit_out = OUT_DIR / "challenge_channel_unit_predictions_v2.csv"
scenario_out = OUT_DIR / "challenge_channel_scenarios_v2.csv"
dashboard_out = OUT_DIR / "challenge_channel_dashboard_summary_v2.csv"
params_out = OUT_DIR / "challenge_channel_best_params_v2.json"
summary_out = OUT_DIR / "challenge_channel_sweep_summary_v2.txt"

res.to_csv(leaderboard_out, index=False)
group_pred.to_csv(group_out, index=False)
unit_pred.to_csv(unit_out, index=False)
best_scen_df.to_csv(scenario_out, index=False)
dashboard_df.to_csv(dashboard_out, index=False)

with open(params_out, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

lines = []
lines.append("CHALLENGE CHANNEL SWEEP V2 CONSTRAINED")
lines.append("=========================")
lines.append("")
lines.append("Best params:")
for k, v in best_params.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("Best metrics:")
for k in [
    "exact_match_rate","precision","recall","specificity","accuracy",
    "tp","fp","fn","tn","red_rate","concern_rate","objective",
    "min_target_recall","mean_target_recall","baseline_concern_rate",
    "baseline_specificity","lameness_recall","ventilation_recall","heat_recall"
]:
    lines.append(f"- {k}: {best_metrics[k]}")
lines.append("")
lines.append("Scenario breakdown:")
lines.append(best_scen_df.to_string(index=False))
lines.append("")
lines.append("Dashboard summary:")
lines.append(dashboard_df.head(40).to_string(index=False))
summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CHALLENGE CHANNEL SWEEP V2 CONSTRAINED ===")
print("\nBest params:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nBest metrics:")
for k in [
    "exact_match_rate","precision","recall","specificity","accuracy",
    "tp","fp","fn","tn","red_rate","concern_rate","objective",
    "min_target_recall","mean_target_recall","baseline_concern_rate",
    "baseline_specificity","lameness_recall","ventilation_recall","heat_recall"
]:
    print(f"{k}: {best_metrics[k]}")

print("\nScenario breakdown:")
print(best_scen_df.to_string(index=False))

print("\nDashboard summary (head):")
print(dashboard_df.head(40).to_string(index=False))

print(f"\nSaved:")
print(f"- {leaderboard_out}")
print(f"- {group_out}")
print(f"- {unit_out}")
print(f"- {scenario_out}")
print(f"- {dashboard_out}")
print(f"- {params_out}")
print(f"- {summary_out}")
