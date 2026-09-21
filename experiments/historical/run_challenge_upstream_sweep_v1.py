import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SIM_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_upstream_sweep_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_IN = SIM_DIR / "animal_states_challenge_sim_v1.csv"

RANDOM_SEED = 2051
N_RANDOM = 1200
TOP_SEEDS = 80
N_REFINE = 2400
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
            out[i] = "WATCH" if scen == "post_event_recovery" else "WATCH"
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

def objective(m):
    score = (
        3.4 * m["recall"] +
        1.4 * m["precision"] +
        1.2 * m["specificity"] +
        0.8 * m["accuracy"] +
        0.5 * m["exact_match_rate"]
    )

    # penalize absurd over-alerting
    if m["concern_rate"] > 0.55:
        score -= 2.0 * (m["concern_rate"] - 0.55)
    if m["red_rate"] > 0.30:
        score -= 2.5 * (m["red_rate"] - 0.30)

    # penalize degenerate no-red regime a bit
    if m["red_rate"] == 0:
        score -= 0.15

    return float(score)

def normalize_params(p):
    p = dict(p)
    p["base_gain"] = float(np.clip(p["base_gain"], 0.8, 2.2))
    p["heat_w"] = float(np.clip(p["heat_w"], 0.0, 0.40))
    p["loco_w"] = float(np.clip(p["loco_w"], 0.0, 0.45))
    p["resp_w"] = float(np.clip(p["resp_w"], 0.0, 0.35))
    p["manage_w"] = float(np.clip(p["manage_w"], 0.0, 0.30))
    p["intake_w"] = float(np.clip(p["intake_w"], 0.0, 0.30))
    p["water_w"] = float(np.clip(p["water_w"], 0.0, 0.25))
    p["visual_w"] = float(np.clip(p["visual_w"], 0.0, 0.30))

    p["animal_y"] = float(np.clip(p["animal_y"], 0.18, 0.60))
    p["animal_r"] = float(np.clip(p["animal_r"], 0.35, 0.85))
    if p["animal_r"] <= p["animal_y"]:
        p["animal_r"] = min(0.85, p["animal_y"] + 0.08)

    p["group_y_share"] = float(np.clip(p["group_y_share"], 0.05, 0.40))
    p["group_r_share"] = float(np.clip(p["group_r_share"], 0.08, 0.50))
    if p["group_r_share"] < p["group_y_share"]:
        p["group_r_share"] = p["group_y_share"] + 0.03

    p["group_y_score"] = float(np.clip(p["group_y_score"], 0.20, 0.70))
    p["group_r_score"] = float(np.clip(p["group_r_score"], 0.30, 0.90))
    if p["group_r_score"] <= p["group_y_score"]:
        p["group_r_score"] = min(0.90, p["group_y_score"] + 0.06)

    p["unit_y_share"] = float(np.clip(p["unit_y_share"], 0.05, 0.50))
    p["unit_r_share"] = float(np.clip(p["unit_r_share"], 0.08, 0.60))
    if p["unit_r_share"] < p["unit_y_share"]:
        p["unit_r_share"] = p["unit_y_share"] + 0.03

    p["unit_y_score"] = float(np.clip(p["unit_y_score"], 0.20, 0.80))
    p["unit_r_score"] = float(np.clip(p["unit_r_score"], 0.30, 0.95))
    if p["unit_r_score"] <= p["unit_y_score"]:
        p["unit_r_score"] = min(0.95, p["unit_y_score"] + 0.06)

    p["min_red_duration"] = int(np.clip(round(p["min_red_duration"]), 1, 4))
    p["min_yellow_duration"] = int(np.clip(round(p["min_yellow_duration"]), 1, 4))
    return p

def sample_params():
    return normalize_params({
        "base_gain": random.uniform(0.9, 1.9),
        "heat_w": random.uniform(0.00, 0.25),
        "loco_w": random.uniform(0.00, 0.30),
        "resp_w": random.uniform(0.00, 0.20),
        "manage_w": random.uniform(0.00, 0.18),
        "intake_w": random.uniform(0.00, 0.18),
        "water_w": random.uniform(0.00, 0.15),
        "visual_w": random.uniform(0.00, 0.18),

        "animal_y": random.uniform(0.22, 0.46),
        "animal_r": random.uniform(0.38, 0.72),

        "group_y_share": random.uniform(0.06, 0.25),
        "group_r_share": random.uniform(0.10, 0.35),
        "group_y_score": random.uniform(0.26, 0.52),
        "group_r_score": random.uniform(0.36, 0.66),

        "unit_y_share": random.uniform(0.06, 0.25),
        "unit_r_share": random.uniform(0.10, 0.35),
        "unit_y_score": random.uniform(0.26, 0.52),
        "unit_r_score": random.uniform(0.36, 0.66),

        "min_red_duration": random.choice([1, 2, 3]),
        "min_yellow_duration": random.choice([1, 2, 3, 4]),
    })

def jitter_params(seed):
    return normalize_params({
        "base_gain": seed["base_gain"] + random.uniform(-0.15, 0.15),
        "heat_w": seed["heat_w"] + random.uniform(-0.04, 0.04),
        "loco_w": seed["loco_w"] + random.uniform(-0.04, 0.04),
        "resp_w": seed["resp_w"] + random.uniform(-0.03, 0.03),
        "manage_w": seed["manage_w"] + random.uniform(-0.03, 0.03),
        "intake_w": seed["intake_w"] + random.uniform(-0.03, 0.03),
        "water_w": seed["water_w"] + random.uniform(-0.02, 0.02),
        "visual_w": seed["visual_w"] + random.uniform(-0.03, 0.03),

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

# ---------------------------------------------------------
# LOAD BASE DATA
# ---------------------------------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df = safe_read(ANIMAL_IN)

needed = [
    "run_id","scenario","hour","unit_id","group_id","animal_id",
    "animal_score","thermal_discomfort","locomotion_quality",
    "respiration_load","management_disruption",
    "feeding_engagement","water_status","visual_anomaly_proxy"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in animal simulation file: {missing}")

df = df[needed].copy()

for c in [
    "animal_score","thermal_discomfort","locomotion_quality",
    "respiration_load","management_disruption",
    "feeding_engagement","water_status","visual_anomaly_proxy"
]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

hours_total = int(df["hour"].max()) + 1

# stable sort
df = df.sort_values(["run_id","scenario","unit_id","group_id","animal_id","hour"]).reset_index(drop=True)

# codes for fast aggregation
group_keys = pd.factorize(
    pd.MultiIndex.from_frame(df[["run_id", "scenario", "hour", "unit_id", "group_id"]])
)[0]
unit_keys = pd.factorize(
    pd.MultiIndex.from_frame(df[["run_id", "scenario", "hour", "unit_id"]])
)[0]

group_count = np.bincount(group_keys)
n_groups = int(group_keys.max()) + 1
n_units = int(unit_keys.max()) + 1

# map each group_key to its unit_key
group_to_unit = np.zeros(n_groups, dtype=np.int64)
tmp_map = pd.DataFrame({"group_key": group_keys, "unit_key": unit_keys}).drop_duplicates("group_key")
group_to_unit[tmp_map["group_key"].to_numpy()] = tmp_map["unit_key"].to_numpy()

# number of groups per unit-time
group_unit_counts = np.bincount(group_to_unit, minlength=n_units)

# unit ordering for persistence
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
unit_order_idx = pd.factorize(
    pd.MultiIndex.from_frame(unit_meta[["run_id", "scenario", "unit_id"]])
)[0]
unit_idx_from_meta = pd.factorize(
    pd.MultiIndex.from_frame(unit_meta[["run_id", "scenario", "hour", "unit_id"]])
)[0]

if not np.array_equal(unit_idx_from_meta, np.arange(len(unit_meta))):
    # enforce identity map
    unit_meta = unit_meta.reset_index(drop=True)
    unit_idx_from_meta = np.arange(len(unit_meta))

seq_positions = []
for _, g in unit_meta.groupby(["run_id","scenario","unit_id"], sort=False):
    seq_positions.append(g.index.to_numpy())

# base arrays
base_score = df["animal_score"].to_numpy(dtype=float)
heat = df["thermal_discomfort"].to_numpy(dtype=float)
loco_drop = (1.0 - df["locomotion_quality"].to_numpy(dtype=float))
resp = df["respiration_load"].to_numpy(dtype=float)
manage = df["management_disruption"].to_numpy(dtype=float)
intake_drop = (1.0 - df["feeding_engagement"].to_numpy(dtype=float))
water_drop = (1.0 - df["water_status"].to_numpy(dtype=float))
visual = df["visual_anomaly_proxy"].to_numpy(dtype=float)

group_info = (
    df[["run_id","scenario","hour","unit_id","group_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
group_info["group_key"] = np.arange(len(group_info))
group_info = group_info.sort_values("group_key").reset_index(drop=True)

# ---------------------------------------------------------
# ENGINE
# ---------------------------------------------------------
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

def evaluate_params(params):
    boosted = clamp01(
        params["base_gain"] * base_score +
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

    # group aggregates
    g_sum = np.bincount(group_keys, weights=boosted, minlength=n_groups)
    g_mean = g_sum / group_count
    g_yshare = np.bincount(group_keys, weights=animal_y.astype(float), minlength=n_groups) / group_count
    g_rshare = np.bincount(group_keys, weights=animal_r.astype(float), minlength=n_groups) / group_count

    group_rank = np.zeros(n_groups, dtype=np.int8)
    red_mask = (g_rshare >= params["group_r_share"]) | (g_mean >= params["group_r_score"])
    y_mask = (g_yshare >= params["group_y_share"]) | (g_mean >= params["group_y_score"])
    group_rank[y_mask] = 2
    group_rank[red_mask] = 3

    # unit aggregates
    u_group_n = group_unit_counts
    u_score_mean = np.bincount(group_to_unit, weights=g_mean, minlength=n_units) / u_group_n
    u_yshare = np.bincount(group_to_unit, weights=(group_rank >= 2).astype(float), minlength=n_units) / u_group_n
    u_rshare = np.bincount(group_to_unit, weights=(group_rank == 3).astype(float), minlength=n_units) / u_group_n

    unit_rank = np.zeros(n_units, dtype=np.int8)
    u_red = (u_rshare >= params["unit_r_share"]) | (u_score_mean >= params["unit_r_score"])
    u_yel = (u_yshare >= params["unit_y_share"]) | (u_score_mean >= params["unit_y_score"])
    unit_rank[u_yel] = 2
    unit_rank[u_red] = 3

    unit_sev = np.array([RANK2SEV[int(x)] for x in unit_rank], dtype=object)

    # persistence per run-scenario-unit trajectory
    for idxs in seq_positions:
        unit_sev[idxs] = apply_persistence(
            unit_sev[idxs],
            params["min_red_duration"],
            params["min_yellow_duration"]
        )

    m = metrics(unit_sev, unit_truth)
    score = objective(m)

    row = dict(params)
    row.update(m)
    row["objective"] = score
    return row, unit_sev, group_rank, g_mean, g_yshare, g_rshare

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------
t0 = time.time()
rows = []
best_payload = None
best_obj = -1e18

print("\n=== CHALLENGE UPSTREAM SWEEP V1 ===")
print(f"animal rows: {len(df)}")
print(f"group slots: {n_groups}")
print(f"unit slots : {n_units}")
print(f"random iters: {N_RANDOM}")
print(f"refine iters: {N_REFINE}")
print("")

seed_rows = []

for i in range(N_RANDOM):
    p = sample_params()
    row, unit_sev, group_rank, g_mean, g_yshare, g_rshare = evaluate_params(p)
    rows.append(row)

    if row["objective"] > best_obj:
        best_obj = row["objective"]
        best_payload = (p, unit_sev.copy(), group_rank.copy(), g_mean.copy(), g_yshare.copy(), g_rshare.copy())

    if (i + 1) % PROGRESS_EVERY == 0:
        print(f"random {i+1}/{N_RANDOM} | elapsed {(time.time()-t0)/60:.1f} min | best {best_obj:.6f}")

seed_df = pd.DataFrame(rows).sort_values(
    ["objective","recall","precision","specificity","accuracy"],
    ascending=False
).reset_index(drop=True)
seed_rows = seed_df.head(TOP_SEEDS).to_dict("records")

for i in range(N_REFINE):
    if seed_rows and random.random() < 0.80:
        p = jitter_params(random.choice(seed_rows[:50]))
    else:
        p = sample_params()

    row, unit_sev, group_rank, g_mean, g_yshare, g_rshare = evaluate_params(p)
    rows.append(row)

    if row["objective"] > best_obj:
        best_obj = row["objective"]
        best_payload = (p, unit_sev.copy(), group_rank.copy(), g_mean.copy(), g_yshare.copy(), g_rshare.copy())

    if (i + 1) % PROGRESS_EVERY == 0:
        print(f"refine {i+1}/{N_REFINE} | elapsed {(time.time()-t0)/60:.1f} min | best {best_obj:.6f}")

res = pd.DataFrame(rows).sort_values(
    ["objective","recall","precision","specificity","accuracy"],
    ascending=False
).reset_index(drop=True)

best_params, best_unit_sev, best_group_rank, best_g_mean, best_g_yshare, best_g_rshare = best_payload
best_metrics = res.iloc[0].to_dict()

# ---------------------------------------------------------
# SAVE BEST PREDICTIONS
# ---------------------------------------------------------
group_pred = group_info.copy()
group_pred["group_score_recal"] = best_g_mean
group_pred["group_yshare_recal"] = best_g_yshare
group_pred["group_rshare_recal"] = best_g_rshare
group_pred["group_severity_recal"] = [RANK2SEV[int(x)] for x in best_group_rank]

unit_pred = unit_meta.copy()
unit_pred["pred_severity_recal"] = best_unit_sev
unit_pred["truth_severity"] = unit_truth
unit_pred["match"] = (unit_pred["pred_severity_recal"] == unit_pred["truth_severity"]).astype(int)

scenario_rows = []
for scen, g in unit_pred.groupby("scenario"):
    m = metrics(g["pred_severity_recal"], g["truth_severity"])
    row = {"scenario": scen}
    row.update(m)
    scenario_rows.append(row)
scenario_df = pd.DataFrame(scenario_rows).sort_values("scenario").reset_index(drop=True)

leaderboard_out = OUT_DIR / "challenge_upstream_sweep_leaderboard_v1.csv"
group_out = OUT_DIR / "challenge_upstream_sweep_group_predictions_v1.csv"
unit_out = OUT_DIR / "challenge_upstream_sweep_unit_predictions_v1.csv"
scenario_out = OUT_DIR / "challenge_upstream_sweep_scenarios_v1.csv"
params_out = OUT_DIR / "challenge_upstream_sweep_best_params_v1.json"
summary_out = OUT_DIR / "challenge_upstream_sweep_summary_v1.txt"

res.to_csv(leaderboard_out, index=False)
group_pred.to_csv(group_out, index=False)
unit_pred.to_csv(unit_out, index=False)
scenario_df.to_csv(scenario_out, index=False)

with open(params_out, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

lines = []
lines.append("CHALLENGE UPSTREAM SWEEP V1")
lines.append("===========================")
lines.append("")
lines.append("Best params:")
for k, v in best_params.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("Best metrics:")
for k in ["exact_match_rate","precision","recall","specificity","accuracy","tp","fp","fn","tn","red_rate","concern_rate","objective"]:
    lines.append(f"- {k}: {best_metrics[k]}")
lines.append("")
lines.append("Scenario breakdown:")
lines.append(scenario_df.to_string(index=False))
summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CHALLENGE UPSTREAM SWEEP V1 ===")
print("\nBest params:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nBest metrics:")
for k in ["exact_match_rate","precision","recall","specificity","accuracy","tp","fp","fn","tn","red_rate","concern_rate","objective"]:
    print(f"{k}: {best_metrics[k]}")

print("\nScenario breakdown:")
print(scenario_df.to_string(index=False))

print(f"\nSaved:")
print(f"- {leaderboard_out}")
print(f"- {group_out}")
print(f"- {unit_out}")
print(f"- {scenario_out}")
print(f"- {params_out}")
print(f"- {summary_out}")

