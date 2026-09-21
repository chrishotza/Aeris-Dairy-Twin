import json
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SIM_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_threshold_sweep_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_IN = SIM_DIR / "group_states_challenge_sim_v1.csv"
UNIT_IN  = SIM_DIR / "unit_states_challenge_sim_v1.csv"

SEV2RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
RANK2SEV = {v: k for k, v in SEV2RANK.items()}

def safe_read(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

group_df = safe_read(GROUP_IN)
unit_df = safe_read(UNIT_IN)

# ---------------------------------------------------------
# Ground truth proxy aligned with the simulator design
# ---------------------------------------------------------
HOURS = int(unit_df["hour"].max()) + 1

def derive_truth(scenario, hour):
    if scenario == "stable_baseline":
        return "GREEN"
    if hour < int(HOURS * 0.35):
        return "GREEN"
    if int(HOURS * 0.35) <= hour < int(HOURS * 0.50):
        return "YELLOW"
    if int(HOURS * 0.50) <= hour < int(HOURS * 0.78):
        return "RED" if scenario != "post_event_recovery" else "YELLOW"
    if scenario == "post_event_recovery":
        return "WATCH"
    return "WATCH"

unit_df = unit_df.copy()
unit_df["truth_severity"] = [derive_truth(s, int(h)) for s, h in zip(unit_df["scenario"], unit_df["hour"])]

# ---------------------------------------------------------
# Rebuild unit severity from group states with sweep params
# ---------------------------------------------------------
def metrics(pred, truth):
    pred_rank = pred.astype(str).str.upper().map(SEV2RANK).fillna(0).astype(int)
    true_rank = truth.astype(str).str.upper().map(SEV2RANK).fillna(0).astype(int)

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
    # seq is list of severity strings
    out = seq[:]

    # suppress isolated RED/YELLOW too short
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
            for k in range(i, j):
                out[k] = "WATCH" if cur == "RED" else "GREEN"
        i = j
    return out

def unit_from_groups(df_g, group_yellow_share, group_red_share, score_yellow, score_red):
    red_share = (df_g["group_severity"] == "RED").mean()
    yellow_share = ((df_g["group_severity"] == "YELLOW") | (df_g["group_severity"] == "RED")).mean()
    mean_score = df_g["group_score"].mean()

    if red_share >= group_red_share or mean_score >= score_red:
        return "RED"
    if yellow_share >= group_yellow_share or mean_score >= score_yellow:
        return "YELLOW"
    return "GREEN"

param_grid = {
    "group_yellow_share": [0.08, 0.10, 0.12, 0.15, 0.18],
    "group_red_share":    [0.12, 0.15, 0.18, 0.22, 0.25],
    "score_yellow":       [0.30, 0.34, 0.38, 0.42],
    "score_red":          [0.42, 0.46, 0.50, 0.55, 0.60],
    "min_red_duration":   [1, 2, 3],
    "min_yellow_duration":[1, 2, 3],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*(param_grid[k] for k in keys)))

rows = []
t0 = time.time()

for idx, combo in enumerate(combos, start=1):
    params = dict(zip(keys, combo))

    # constraints
    if params["group_red_share"] < params["group_yellow_share"]:
        continue
    if params["score_red"] <= params["score_yellow"]:
        continue

    pred_rows = []
    for (run_id, unit_id, scenario), g in group_df.groupby(["run_id", "unit_id", "scenario"]):
        g = g.sort_values("hour").reset_index(drop=True)

        raw_seq = []
        for hour, gh in g.groupby("hour"):
            sev = unit_from_groups(
                gh,
                group_yellow_share=params["group_yellow_share"],
                group_red_share=params["group_red_share"],
                score_yellow=params["score_yellow"],
                score_red=params["score_red"],
            )
            raw_seq.append((int(hour), sev))

        hours = [h for h, s in raw_seq]
        raw_sev = [s for h, s in raw_seq]
        adj_sev = apply_persistence(
            raw_sev,
            min_red_duration=params["min_red_duration"],
            min_yellow_duration=params["min_yellow_duration"],
        )

        for h, sev in zip(hours, adj_sev):
            pred_rows.append({
                "run_id": run_id,
                "unit_id": unit_id,
                "scenario": scenario,
                "hour": h,
                "pred_severity": sev,
            })

    pred_df = pd.DataFrame(pred_rows)
    eval_df = pred_df.merge(
        unit_df[["run_id", "unit_id", "scenario", "hour", "truth_severity"]],
        on=["run_id", "unit_id", "scenario", "hour"],
        how="left"
    )

    m = metrics(eval_df["pred_severity"], eval_df["truth_severity"])

    # objective: raise recall strongly, keep precision/specificity acceptable
    score = (
        3.0 * m["recall"] +
        1.4 * m["precision"] +
        1.2 * m["specificity"] +
        0.8 * m["accuracy"] +
        0.5 * m["exact_match_rate"]
    )

    # penalize absurd over-alerting
    if m["concern_rate"] > 0.45:
        score -= 2.0 * (m["concern_rate"] - 0.45)
    if m["red_rate"] > 0.20:
        score -= 2.5 * (m["red_rate"] - 0.20)

    row = dict(params)
    row.update(m)
    row["objective"] = score
    rows.append(row)

    if idx % 100 == 0:
        print(f"tested {idx}/{len(combos)} | elapsed {(time.time()-t0)/60:.1f} min")

res = pd.DataFrame(rows).sort_values(
    ["objective", "recall", "precision", "specificity", "accuracy"],
    ascending=False
).reset_index(drop=True)

best = res.iloc[0].to_dict()

# rebuild best prediction set
best_pred_rows = []
for (run_id, unit_id, scenario), g in group_df.groupby(["run_id", "unit_id", "scenario"]):
    g = g.sort_values("hour").reset_index(drop=True)

    raw_seq = []
    for hour, gh in g.groupby("hour"):
        sev = unit_from_groups(
            gh,
            group_yellow_share=best["group_yellow_share"],
            group_red_share=best["group_red_share"],
            score_yellow=best["score_yellow"],
            score_red=best["score_red"],
        )
        raw_seq.append((int(hour), sev))

    hours = [h for h, s in raw_seq]
    raw_sev = [s for h, s in raw_seq]
    adj_sev = apply_persistence(
        raw_sev,
        min_red_duration=int(best["min_red_duration"]),
        min_yellow_duration=int(best["min_yellow_duration"]),
    )

    for h, sev in zip(hours, adj_sev):
        best_pred_rows.append({
            "run_id": run_id,
            "unit_id": unit_id,
            "scenario": scenario,
            "hour": h,
            "pred_severity": sev,
        })

best_pred_df = pd.DataFrame(best_pred_rows).merge(
    unit_df[["run_id", "unit_id", "scenario", "hour", "truth_severity"]],
    on=["run_id", "unit_id", "scenario", "hour"],
    how="left"
)

# scenario-level breakdown
scen_rows = []
for scen, g in best_pred_df.groupby("scenario"):
    m = metrics(g["pred_severity"], g["truth_severity"])
    row = {"scenario": scen}
    row.update(m)
    scen_rows.append(row)
scen_df = pd.DataFrame(scen_rows).sort_values("scenario").reset_index(drop=True)

leaderboard_out = OUT_DIR / "challenge_threshold_sweep_leaderboard_v1.csv"
best_pred_out   = OUT_DIR / "challenge_threshold_sweep_best_predictions_v1.csv"
scenario_out    = OUT_DIR / "challenge_threshold_sweep_scenarios_v1.csv"
summary_out     = OUT_DIR / "challenge_threshold_sweep_summary_v1.txt"
params_out      = OUT_DIR / "challenge_threshold_sweep_best_params_v1.json"

res.to_csv(leaderboard_out, index=False)
best_pred_df.to_csv(best_pred_out, index=False)
scen_df.to_csv(scenario_out, index=False)

with open(params_out, "w", encoding="utf-8") as f:
    json.dump({
        "group_yellow_share": best["group_yellow_share"],
        "group_red_share": best["group_red_share"],
        "score_yellow": best["score_yellow"],
        "score_red": best["score_red"],
        "min_red_duration": int(best["min_red_duration"]),
        "min_yellow_duration": int(best["min_yellow_duration"]),
    }, f, indent=2)

lines = []
lines.append("CHALLENGE THRESHOLD SWEEP V1")
lines.append("============================")
lines.append("")
lines.append("Best params:")
for k in ["group_yellow_share","group_red_share","score_yellow","score_red","min_red_duration","min_yellow_duration"]:
    lines.append(f"- {k}: {best[k]}")
lines.append("")
lines.append("Best metrics:")
for k in ["exact_match_rate","precision","recall","specificity","accuracy","tp","fp","fn","tn","red_rate","concern_rate","objective"]:
    lines.append(f"- {k}: {best[k]}")
lines.append("")
lines.append("Scenario breakdown:")
lines.append(scen_df.to_string(index=False))

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CHALLENGE THRESHOLD SWEEP V1 ===")
print("\nBest params:")
for k in ["group_yellow_share","group_red_share","score_yellow","score_red","min_red_duration","min_yellow_duration"]:
    print(f"{k}: {best[k]}")

print("\nBest metrics:")
for k in ["exact_match_rate","precision","recall","specificity","accuracy","tp","fp","fn","tn","red_rate","concern_rate","objective"]:
    print(f"{k}: {best[k]}")

print("\nScenario breakdown:")
print(scen_df.to_string(index=False))

print(f"\nSaved:")
print(f"- {leaderboard_out}")
print(f"- {best_pred_out}")
print(f"- {scenario_out}")
print(f"- {params_out}")
print(f"- {summary_out}")

