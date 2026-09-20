import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SIM_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
CHAMP_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_sim_reinforced_v2"
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_ventilation_refinement_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_IN = SIM_DIR / "animal_states_challenge_sim_v1.csv"
CHAMP_PARAMS_IN = CHAMP_DIR / "challenge_sim_reinforced_best_params_v2.json"

RANDOM_SEED = 2097
N_RANDOM = 800
TOP_SEEDS = 80
N_REFINE = 1600
PROGRESS_EVERY = 100

SEV2RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
RANK2SEV = {v: k for k, v in SEV2RANK.items()}

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def safe_read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def safe_read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

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

df = safe_read_csv(ANIMAL_IN)
champ = safe_read_json(CHAMP_PARAMS_IN)

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

scenario_arr = df["scenario"].astype(str).to_numpy()
phase = clamp01(df["phase"].to_numpy(dtype=float))

base_score = df["animal_score"].to_numpy(dtype=float)
heat = df["thermal_discomfort"].to_numpy(dtype=float)
loco_drop = (1.0 - df["locomotion_quality"].to_numpy(dtype=float))
resp = df["respiration_load"].to_numpy(dtype=float)
manage = df["management_disruption"].to_numpy(dtype=float)
intake_drop = (1.0 - df["feeding_engagement"].to_numpy(dtype=float))
water_drop = (1.0 - df["water_status"].to_numpy(dtype=float))
visual = df["visual_anomaly_proxy"].to_numpy(dtype=float)

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

vent_focus = clamp01(
    0.34 * (1.0 - df["ventilation_quality"].to_numpy(dtype=float)) +
    0.26 * df["respiration_load"].to_numpy(dtype=float) +
    0.18 * df["thermal_discomfort"].to_numpy(dtype=float) +
    0.10 * df["visual_anomaly_proxy"].to_numpy(dtype=float) +
    0.12 * (1.0 - df["activity"].to_numpy(dtype=float))
)

is_heat = (scenario_arr == "heat_stress_wave").astype(float)
is_lame = (scenario_arr == "lameness_cluster").astype(float)
is_vent = (scenario_arr == "ventilation_failure").astype(float)

group_info = (
    df[["run_id","scenario","hour","unit_id","group_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
group_info["group_key"] = np.arange(len(group_info))
group_info = group_info.sort_values("group_key").reset_index(drop=True)

# ---------------------------------------------------------
# PARAMS
# ---------------------------------------------------------
def normalize_params(p):
    p = dict(p)

    p["vent_amp"] = float(np.clip(p["vent_amp"], 0.0, 1.20))
    p["resp_w"] = float(np.clip(p["resp_w"], 0.08, 0.30))
    p["manage_w"] = float(np.clip(p["manage_w"], 0.08, 0.24))
    p["visual_w"] = float(np.clip(p["visual_w"], 0.04, 0.18))
    p["base_gain"] = float(np.clip(p["base_gain"], 0.58, 0.78))

    p["group_y_share"] = float(np.clip(p["group_y_share"], 0.09, 0.18))
    p["group_r_share"] = float(np.clip(p["group_r_share"], 0.22, 0.34))
    p["group_y_score"] = float(np.clip(p["group_y_score"], 0.34, 0.42))
    p["group_r_score"] = float(np.clip(p["group_r_score"], 0.48, 0.58))

    p["unit_y_share"] = float(np.clip(p["unit_y_share"], 0.06, 0.12))
    p["unit_r_share"] = float(np.clip(p["unit_r_share"], 0.22, 0.34))
    p["unit_y_score"] = float(np.clip(p["unit_y_score"], 0.46, 0.54))
    p["unit_r_score"] = float(np.clip(p["unit_r_score"], 0.50, 0.58))

    p["animal_y"] = float(np.clip(p["animal_y"], 0.42, 0.49))
    p["animal_r"] = float(np.clip(p["animal_r"], 0.54, 0.62))
    if p["animal_r"] <= p["animal_y"]:
        p["animal_r"] = min(0.62, p["animal_y"] + 0.08)

    p["min_red_duration"] = int(np.clip(round(p["min_red_duration"]), 1, 2))
    p["min_yellow_duration"] = int(np.clip(round(p["min_yellow_duration"]), 3, 4))
    return p

def sample_params():
    return normalize_params({
        "vent_amp": random.uniform(0.15, 0.85),
        "resp_w": champ["resp_w"] + random.uniform(-0.04, 0.05),
        "manage_w": champ["manage_w"] + random.uniform(-0.03, 0.03),
        "visual_w": champ["visual_w"] + random.uniform(-0.03, 0.03),
        "base_gain": champ["base_gain"] + random.uniform(-0.06, 0.06),

        "animal_y": champ["animal_y"] + random.uniform(-0.02, 0.02),
        "animal_r": champ["animal_r"] + random.uniform(-0.02, 0.02),

        "group_y_share": champ["group_y_share"] + random.uniform(-0.02, 0.02),
        "group_r_share": champ["group_r_share"] + random.uniform(-0.03, 0.03),
        "group_y_score": champ["group_y_score"] + random.uniform(-0.03, 0.03),
        "group_r_score": champ["group_r_score"] + random.uniform(-0.03, 0.03),

        "unit_y_share": champ["unit_y_share"] + random.uniform(-0.02, 0.02),
        "unit_r_share": champ["unit_r_share"] + random.uniform(-0.03, 0.03),
        "unit_y_score": champ["unit_y_score"] + random.uniform(-0.02, 0.02),
        "unit_r_score": champ["unit_r_score"] + random.uniform(-0.02, 0.02),

        "min_red_duration": champ["min_red_duration"] + random.choice([-1, 0, 1]),
        "min_yellow_duration": champ["min_yellow_duration"] + random.choice([-1, 0, 1]),
    })

def jitter_params(seed):
    return normalize_params({
        "vent_amp": seed["vent_amp"] + random.uniform(-0.08, 0.08),
        "resp_w": seed["resp_w"] + random.uniform(-0.02, 0.02),
        "manage_w": seed["manage_w"] + random.uniform(-0.015, 0.015),
        "visual_w": seed["visual_w"] + random.uniform(-0.015, 0.015),
        "base_gain": seed["base_gain"] + random.uniform(-0.03, 0.03),

        "animal_y": seed["animal_y"] + random.uniform(-0.015, 0.015),
        "animal_r": seed["animal_r"] + random.uniform(-0.015, 0.015),

        "group_y_share": seed["group_y_share"] + random.uniform(-0.01, 0.01),
        "group_r_share": seed["group_r_share"] + random.uniform(-0.015, 0.015),
        "group_y_score": seed["group_y_score"] + random.uniform(-0.015, 0.015),
        "group_r_score": seed["group_r_score"] + random.uniform(-0.015, 0.015),

        "unit_y_share": seed["unit_y_share"] + random.uniform(-0.01, 0.01),
        "unit_r_share": seed["unit_r_share"] + random.uniform(-0.015, 0.015),
        "unit_y_score": seed["unit_y_score"] + random.uniform(-0.015, 0.015),
        "unit_r_score": seed["unit_r_score"] + random.uniform(-0.015, 0.015),

        "min_red_duration": seed["min_red_duration"] + random.choice([-1, 0, 1]),
        "min_yellow_duration": seed["min_yellow_duration"] + random.choice([-1, 0, 1]),
    })

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
def evaluate_params(params):
    reinforced_base = clamp01(
        base_score +
        champ["heat_amp"] * is_heat * phase * heat_focus +
        champ["lame_amp"] * is_lame * phase * lame_focus +
        params["vent_amp"] * is_vent * phase * vent_focus
    )

    boosted = clamp01(
        params["base_gain"] * reinforced_base +
        champ["heat_w"] * heat +
        champ["loco_w"] * loco_drop +
        params["resp_w"] * resp +
        params["manage_w"] * manage +
        champ["intake_w"] * intake_drop +
        champ["water_w"] * water_drop +
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
    heatm = scen_df[scen_df["scenario"] == "heat_stress_wave"].iloc[0]
    vent = scen_df[scen_df["scenario"] == "ventilation_failure"].iloc[0]
    feedm = scen_df[scen_df["scenario"] == "feeding_disruption"].iloc[0]
    waterm = scen_df[scen_df["scenario"] == "water_system_issue"].iloc[0]

    # reference floors from current champion
    target_vent = 0.44512195121951187
    target_heat = 0.9810298102981023
    target_lame = 0.9756097560975604
    target_feed = 0.9397018970189696
    target_water = 0.9550304878048774
    target_baseline_concern = 0.019675925925925927
    target_baseline_spec = 0.9803240740740738

    score = (
        2.8 * float(vent["recall"]) +
        0.9 * float(overall["precision"]) +
        0.9 * float(overall["specificity"]) +
        0.8 * float(overall["accuracy"]) +
        0.6 * float(heatm["recall"]) +
        0.6 * float(lame["recall"]) +
        0.3 * float(feedm["recall"]) +
        0.3 * float(waterm["recall"])
    )

    # protect current strengths
    score -= 2.0 * max(0.0, target_heat - float(heatm["recall"]))
    score -= 2.0 * max(0.0, target_lame - float(lame["recall"]))
    score -= 1.2 * max(0.0, target_feed - float(feedm["recall"]))
    score -= 1.2 * max(0.0, target_water - float(waterm["recall"]))

    # keep baseline clean
    score -= 2.2 * max(0.0, float(baseline["concern_rate"]) - target_baseline_concern)
    score -= 1.8 * max(0.0, target_baseline_spec - float(baseline["specificity"]))

    # keep flooding under control
    if float(overall["concern_rate"]) > 0.48:
        score -= 2.0 * (float(overall["concern_rate"]) - 0.48)
    if float(overall["red_rate"]) > 0.10:
        score -= 2.0 * (float(overall["red_rate"]) - 0.10)

    # viability gate
    viable = True
    if float(baseline["concern_rate"]) > 0.05:
        viable = False
    if float(baseline["specificity"]) < 0.95:
        viable = False
    if float(heatm["recall"]) < 0.90:
        viable = False
    if float(lame["recall"]) < 0.88:
        viable = False
    if float(feedm["recall"]) < 0.88:
        viable = False
    if float(waterm["recall"]) < 0.88:
        viable = False
    if float(overall["precision"]) < 0.60:
        viable = False

    if not viable:
        score -= 100.0

    row = dict(params)
    row.update(overall)
    row["objective"] = float(score)
    row["baseline_concern_rate"] = float(baseline["concern_rate"])
    row["baseline_specificity"] = float(baseline["specificity"])
    row["ventilation_recall"] = float(vent["recall"])
    row["heat_recall"] = float(heatm["recall"])
    row["lameness_recall"] = float(lame["recall"])
    row["feed_recall"] = float(feedm["recall"])
    row["water_recall"] = float(waterm["recall"])

    return row, scen_df, unit_sev, group_rank, g_mean, g_yshare, g_rshare

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------
t0 = time.time()
rows = []
best_payload = None
best_obj = -1e18

print("\n=== CHALLENGE VENTILATION REFINEMENT V3 ===")
print(f"animal rows: {len(df)}")
print(f"group slots: {n_groups}")
print(f"unit slots : {n_units}")
print(f"random iters: {N_RANDOM}")
print(f"refine iters: {N_REFINE}")
print("")

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
    ["objective","ventilation_recall","baseline_specificity","precision"],
    ascending=False
).reset_index(drop=True)
seed_rows = seed_df.head(TOP_SEEDS).to_dict("records")

for i in range(N_REFINE):
    if seed_rows and random.random() < 0.82:
        p = jitter_params(random.choice(seed_rows[:50]))
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
    ["objective","ventilation_recall","baseline_specificity","precision"],
    ascending=False
).reset_index(drop=True)

best_params, best_scen_df, best_unit_sev, best_group_rank, best_g_mean, best_g_yshare, best_g_rshare = best_payload
best_metrics = res.iloc[0].to_dict()

group_pred = group_info.copy()
group_pred["group_score_vent_v3"] = best_g_mean
group_pred["group_yshare_vent_v3"] = best_g_yshare
group_pred["group_rshare_vent_v3"] = best_g_rshare
group_pred["group_severity_vent_v3"] = [RANK2SEV[int(x)] for x in best_group_rank]

unit_pred = unit_meta.copy()
unit_pred["pred_severity_vent_v3"] = best_unit_sev
unit_pred["truth_severity"] = unit_truth
unit_pred["match"] = (unit_pred["pred_severity_vent_v3"] == unit_pred["truth_severity"]).astype(int)

dashboard_rows = []
for (scenario, unit_id), g in unit_pred.groupby(["scenario","unit_id"]):
    dashboard_rows.append({
        "scenario": scenario,
        "unit_id": unit_id,
        "hours_total": int(len(g)),
        "hours_green": int((g["pred_severity_vent_v3"] == "GREEN").sum()),
        "hours_yellow": int((g["pred_severity_vent_v3"] == "YELLOW").sum()),
        "hours_red": int((g["pred_severity_vent_v3"] == "RED").sum()),
        "hours_concern": int(g["pred_severity_vent_v3"].isin(["YELLOW","RED"]).sum()),
        "concern_rate": round(float(g["pred_severity_vent_v3"].isin(["YELLOW","RED"]).mean()), 4),
        "red_rate": round(float((g["pred_severity_vent_v3"] == "RED").mean()), 4),
        "match_rate": round(float(g["match"].mean()), 4),
    })
dashboard_df = pd.DataFrame(dashboard_rows).sort_values(["scenario","unit_id"]).reset_index(drop=True)

leaderboard_out = OUT_DIR / "challenge_ventilation_refinement_leaderboard_v3.csv"
group_out = OUT_DIR / "challenge_ventilation_group_predictions_v3.csv"
unit_out = OUT_DIR / "challenge_ventilation_unit_predictions_v3.csv"
scenario_out = OUT_DIR / "challenge_ventilation_scenarios_v3.csv"
dashboard_out = OUT_DIR / "challenge_ventilation_dashboard_v3.csv"
params_out = OUT_DIR / "challenge_ventilation_best_params_v3.json"
summary_out = OUT_DIR / "challenge_ventilation_summary_v3.txt"

res.to_csv(leaderboard_out, index=False)
group_pred.to_csv(group_out, index=False)
unit_pred.to_csv(unit_out, index=False)
best_scen_df.to_csv(scenario_out, index=False)
dashboard_df.to_csv(dashboard_out, index=False)

with open(params_out, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

lines = []
lines.append("CHALLENGE VENTILATION REFINEMENT V3")
lines.append("===================================")
lines.append("")
lines.append("Best params:")
for k, v in best_params.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("Best metrics:")
for k in [
    "exact_match_rate","precision","recall","specificity","accuracy",
    "tp","fp","fn","tn","red_rate","concern_rate","objective",
    "baseline_concern_rate","baseline_specificity","ventilation_recall",
    "heat_recall","lameness_recall","feed_recall","water_recall"
]:
    lines.append(f"- {k}: {best_metrics[k]}")
lines.append("")
lines.append("Scenario breakdown:")
lines.append(best_scen_df.to_string(index=False))
lines.append("")
lines.append("Dashboard summary:")
lines.append(dashboard_df.head(40).to_string(index=False))
summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CHALLENGE VENTILATION REFINEMENT V3 ===")
print("\nBest params:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nBest metrics:")
for k in [
    "exact_match_rate","precision","recall","specificity","accuracy",
    "tp","fp","fn","tn","red_rate","concern_rate","objective",
    "baseline_concern_rate","baseline_specificity","ventilation_recall",
    "heat_recall","lameness_recall","feed_recall","water_recall"
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
