import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# PATHS
# =========================================================

ROOT = Path(".").resolve()
EXP_DIR = ROOT / "11_real_data" / "cvb_data" / "exports"
OUT_DIR = EXP_DIR / "cvb_param_sweep_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V4_IN = EXP_DIR / "cvb_clip_alert_feed_v4_visibility_gated.csv"
CMP_IN = EXP_DIR / "cvb_aeris_vs_validated_reference_v1.csv"
REF_IN = EXP_DIR / "cvb_validated_reference_by_clip_v1.csv"
CALIB_IN = EXP_DIR / "cvb_behavior_code_calibration_v1.csv"

# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 2037
COARSE_RANDOM_ITERS = 5000
REFINE_RANDOM_ITERS = 25000
FINALISTS_FOR_BOOTSTRAP = 100
BOOTSTRAP_N = 200
PROGRESS_EVERY = 1000

MIN_RECALL_FLOOR = 1.0

W_EXACT = 1.30
W_PREC  = 2.30
W_SPEC  = 1.80
W_ACC   = 1.10
W_REC   = 0.90

PENALTY_RECALL = 120.0
PENALTY_RED_RATE = 1.00
PENALTY_CONCERN_RATE = 0.55

SEV2RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
RANK2SEV = {v: k for k, v in SEV2RANK.items()}

# =========================================================
# HELPERS
# =========================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def extract_behavior_code(clip_id: str) -> str:
    import re
    m = re.search(r"beh(\d+)", str(clip_id))
    return m.group(1) if m else "unknown"

def backfill_by_key(base: pd.DataFrame, other: pd.DataFrame, key: str, cols):
    cols = [c for c in cols if c in other.columns]
    if not cols:
        return base
    tmp = other[[key] + cols].drop_duplicates(key).copy()
    tmp = tmp.rename(columns={c: f"{c}__fill" for c in cols})
    base = base.merge(tmp, on=key, how="left")
    for c in cols:
        fill_col = f"{c}__fill"
        if c not in base.columns:
            base[c] = base[fill_col]
        else:
            base[c] = base[c].fillna(base[fill_col])
        base.drop(columns=[fill_col], inplace=True)
    return base

def metrics_from_pred(pred_rank, ref_rank):
    pred_rank = np.asarray(pred_rank, dtype=np.int8)
    ref_rank = np.asarray(ref_rank, dtype=np.int8)

    exact = float((pred_rank == ref_rank).mean())

    pred_concern = (pred_rank >= 2).astype(np.int8)
    ref_concern = (ref_rank >= 2).astype(np.int8)

    tp = int(((pred_concern == 1) & (ref_concern == 1)).sum())
    fp = int(((pred_concern == 1) & (ref_concern == 0)).sum())
    fn = int(((pred_concern == 0) & (ref_concern == 1)).sum())
    tn = int(((pred_concern == 0) & (ref_concern == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    accuracy = (tp + tn) / max(len(pred_rank), 1)

    red_rate = float((pred_rank == 3).mean())
    concern_rate = float((pred_rank >= 2).mean())

    return {
        "exact_match_rate": exact,
        "concern_precision": precision,
        "concern_recall": recall,
        "concern_specificity": specificity,
        "concern_accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "red_rate": red_rate,
        "concern_rate": concern_rate,
    }

def objective(metrics, target_red_rate, target_concern_rate):
    score = (
        W_EXACT * metrics["exact_match_rate"] +
        W_PREC  * metrics["concern_precision"] +
        W_SPEC  * metrics["concern_specificity"] +
        W_ACC   * metrics["concern_accuracy"] +
        W_REC   * metrics["concern_recall"]
    )

    if metrics["concern_recall"] < MIN_RECALL_FLOOR:
        score -= PENALTY_RECALL * (MIN_RECALL_FLOOR - metrics["concern_recall"])

    if metrics["red_rate"] > target_red_rate:
        score -= PENALTY_RED_RATE * (metrics["red_rate"] - target_red_rate)

    if metrics["concern_rate"] > target_concern_rate:
        score -= PENALTY_CONCERN_RATE * (metrics["concern_rate"] - target_concern_rate)

    return float(score)

def normalize_params(p):
    p = dict(p)
    p["min_evidence"] = int(max(5, min(60, round(p["min_evidence"]))))
    p["extreme_eps"] = float(max(0.0, min(0.02, p["extreme_eps"])))
    p["amb_watch"] = float(max(0.10, min(0.70, p["amb_watch"])))
    p["vis_watch"] = float(max(0.30, min(0.85, p["vis_watch"])))
    p["amb_red_cap"] = float(max(0.10, min(0.70, p["amb_red_cap"])))
    p["vis_red_cap"] = float(max(0.30, min(0.90, p["vis_red_cap"])))
    p["amb_yellow_cap"] = float(max(0.10, min(0.70, p["amb_yellow_cap"])))
    p["vis_yellow_cap"] = float(max(0.30, min(0.90, p["vis_yellow_cap"])))
    p["lowrisk_red_score"] = float(max(0.05, min(0.60, p["lowrisk_red_score"])))
    p["lowrisk_yel_score"] = float(max(0.03, min(0.40, p["lowrisk_yel_score"])))
    p["ambig_extra_margin"] = float(max(0.0, min(0.15, p["ambig_extra_margin"])))
    p["moderate_vis_cap"] = float(max(0.30, min(0.85, p["moderate_vis_cap"])))

    p["amb_red_cap"] = min(p["amb_red_cap"], p["amb_watch"])
    p["amb_yellow_cap"] = max(p["amb_red_cap"], p["amb_yellow_cap"])
    p["vis_red_cap"] = max(p["vis_red_cap"], p["vis_watch"])
    p["vis_yellow_cap"] = max(p["vis_watch"], p["vis_yellow_cap"])
    p["lowrisk_yel_score"] = min(p["lowrisk_yel_score"], p["lowrisk_red_score"])

    return p

def sample_random_params():
    return normalize_params({
        "min_evidence": random.choice([5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]),
        "extreme_eps": random.choice([0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01]),
        "amb_watch": random.uniform(0.15, 0.55),
        "vis_watch": random.uniform(0.40, 0.75),
        "amb_red_cap": random.uniform(0.15, 0.45),
        "vis_red_cap": random.uniform(0.45, 0.85),
        "amb_yellow_cap": random.uniform(0.15, 0.55),
        "vis_yellow_cap": random.uniform(0.45, 0.80),
        "lowrisk_red_score": random.uniform(0.15, 0.45),
        "lowrisk_yel_score": random.uniform(0.08, 0.22),
        "ambig_extra_margin": random.uniform(0.00, 0.10),
        "moderate_vis_cap": random.uniform(0.40, 0.75),
    })

def jitter_params(seed):
    return normalize_params({
        "min_evidence": seed["min_evidence"] + random.choice([-8, -5, -2, 0, 2, 5, 8]),
        "extreme_eps": seed["extreme_eps"] + random.choice([-0.0025, -0.001, 0.0, 0.001, 0.0025]),
        "amb_watch": seed["amb_watch"] + random.uniform(-0.06, 0.06),
        "vis_watch": seed["vis_watch"] + random.uniform(-0.06, 0.06),
        "amb_red_cap": seed["amb_red_cap"] + random.uniform(-0.06, 0.06),
        "vis_red_cap": seed["vis_red_cap"] + random.uniform(-0.06, 0.06),
        "amb_yellow_cap": seed["amb_yellow_cap"] + random.uniform(-0.06, 0.06),
        "vis_yellow_cap": seed["vis_yellow_cap"] + random.uniform(-0.06, 0.06),
        "lowrisk_red_score": seed["lowrisk_red_score"] + random.uniform(-0.05, 0.05),
        "lowrisk_yel_score": seed["lowrisk_yel_score"] + random.uniform(-0.03, 0.03),
        "ambig_extra_margin": seed["ambig_extra_margin"] + random.uniform(-0.03, 0.03),
        "moderate_vis_cap": seed["moderate_vis_cap"] + random.uniform(-0.05, 0.05),
    })

# =========================================================
# LOAD
# =========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

main = safe_read_csv(V4_IN)
cmp = safe_read_csv(CMP_IN)
ref = safe_read_csv(REF_IN)
calib = safe_read_csv(CALIB_IN)

if "clip_id" not in main.columns:
    raise ValueError("V4 file must contain clip_id")

if "severity_v4" not in main.columns:
    raise ValueError("V4 file must contain severity_v4")

if "behavior_code" not in main.columns:
    main["behavior_code"] = main["clip_id"].astype(str).apply(extract_behavior_code)

main = backfill_by_key(main, cmp, "clip_id", [
    "validated_ref_severity",
    "validated_ref_reason",
    "p_ambiguous",
    "p_extreme",
    "candidate_score",
])

main = backfill_by_key(main, ref, "clip_id", ["total_obs"])

calib["behavior_code"] = calib["behavior_code"].astype(str)
main["behavior_code"] = main["behavior_code"].astype(str)
main = backfill_by_key(main, calib, "behavior_code", ["calibration_class"])

needed = [
    "clip_id",
    "severity_v4",
    "validated_ref_severity",
    "candidate_score",
    "mean_visibility_proxy",
    "p_ambiguous",
    "p_extreme",
    "total_obs",
    "calibration_class",
]
missing = [c for c in needed if c not in main.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = main.copy()
df["severity_v4"] = df["severity_v4"].astype(str).str.upper()
df["validated_ref_severity"] = df["validated_ref_severity"].astype(str).str.upper()
df["calibration_class"] = df["calibration_class"].fillna("unknown").astype(str)

for col in ["candidate_score", "mean_visibility_proxy", "p_ambiguous", "p_extreme", "total_obs"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

base_rank = df["severity_v4"].map(SEV2RANK).fillna(0).astype(np.int8).to_numpy()
ref_rank = df["validated_ref_severity"].map(SEV2RANK).fillna(0).astype(np.int8).to_numpy()

candidate_score = df["candidate_score"].to_numpy(dtype=float)
vis = df["mean_visibility_proxy"].to_numpy(dtype=float)
p_amb = df["p_ambiguous"].to_numpy(dtype=float)
p_ext = df["p_extreme"].to_numpy(dtype=float)
total_obs = df["total_obs"].to_numpy(dtype=float)

cal_lowrisk = (df["calibration_class"] == "low_risk_code").to_numpy()
cal_ambig = (df["calibration_class"] == "visibility_or_ambiguity_code").to_numpy()
cal_mod = (df["calibration_class"] == "moderate_risk_code").to_numpy()
cal_high = (df["calibration_class"] == "high_risk_code").to_numpy()

ref_metrics = metrics_from_pred(ref_rank, ref_rank)
target_red_rate = max(0.01, ref_metrics["red_rate"] + 0.01)
target_concern_rate = max(0.03, ref_metrics["concern_rate"] + 0.03)

baseline_v4_metrics = metrics_from_pred(base_rank, ref_rank)

# =========================================================
# APPLY PARAMS
# =========================================================

def apply_params_to_rank(params):
    rank = base_rank.copy()
    extreme_zero = (p_ext <= params["extreme_eps"])

    # low evidence => WATCH
    m = extreme_zero & (total_obs < params["min_evidence"])
    rank[m] = np.minimum(rank[m], 1)

    # global ambiguity/visibility gating => WATCH
    m = extreme_zero & ((p_amb >= params["amb_watch"]) | (vis < params["vis_watch"]))
    rank[m] = np.minimum(rank[m], 1)

    # RED cap => WATCH
    m = (rank == 3) & extreme_zero & ((p_amb >= params["amb_red_cap"]) | (vis < params["vis_red_cap"]))
    rank[m] = 1

    # YELLOW cap => WATCH
    m = (rank == 2) & extreme_zero & ((p_amb >= params["amb_yellow_cap"]) | (vis < params["vis_yellow_cap"]))
    rank[m] = 1

    # ambiguity codes => extra caution
    m = cal_ambig & extreme_zero & (p_amb >= max(0.0, params["amb_watch"] - params["ambig_extra_margin"]))
    rank[m] = np.minimum(rank[m], 1)

    # low risk codes => only demote
    m = cal_lowrisk & (rank == 3) & extreme_zero & (candidate_score < params["lowrisk_red_score"])
    rank[m] = 2

    m = cal_lowrisk & (rank == 2) & extreme_zero & (candidate_score < params["lowrisk_yel_score"])
    rank[m] = 1

    # moderate codes => if weak visibility, cap YELLOW -> WATCH
    m = cal_mod & (rank == 2) & extreme_zero & (vis < params["moderate_vis_cap"])
    rank[m] = 1

    # high-risk codes are never auto-promoted
    m = cal_high & (rank == 3) & extreme_zero & (vis < 0.40) & (p_amb > 0.25)
    rank[m] = 2

    return rank

def evaluate_params(params):
    pred = apply_params_to_rank(params)
    m = metrics_from_pred(pred, ref_rank)
    row = dict(params)
    row.update(m)
    row["objective"] = objective(m, target_red_rate, target_concern_rate)
    return row

def bootstrap_eval(params, n_boot=BOOTSTRAP_N):
    pred = apply_params_to_rank(params)
    n = len(pred)

    objs = []
    exacts = []
    precs = []
    recs = []
    specs = []
    accs = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        m = metrics_from_pred(pred[idx], ref_rank[idx])
        objs.append(objective(m, target_red_rate, target_concern_rate))
        exacts.append(m["exact_match_rate"])
        precs.append(m["concern_precision"])
        recs.append(m["concern_recall"])
        specs.append(m["concern_specificity"])
        accs.append(m["concern_accuracy"])

    return {
        "bootstrap_obj_mean": float(np.mean(objs)),
        "bootstrap_obj_std": float(np.std(objs)),
        "bootstrap_exact_mean": float(np.mean(exacts)),
        "bootstrap_precision_mean": float(np.mean(precs)),
        "bootstrap_recall_mean": float(np.mean(recs)),
        "bootstrap_specificity_mean": float(np.mean(specs)),
        "bootstrap_accuracy_mean": float(np.mean(accs)),
    }

# =========================================================
# SEARCH
# =========================================================

start = time.time()
results = []
seen = set()

def key_of(p):
    return tuple(round(float(p[k]), 6) if not isinstance(p[k], int) else int(p[k]) for k in sorted(p.keys()))

print("\n=== CVB PARAM SWEEP V2 ===")
print(f"Rows: {len(df)}")
print(f"Target red rate    : {target_red_rate:.4f}")
print(f"Target concern rate: {target_concern_rate:.4f}")
print(f"Coarse random iters: {COARSE_RANDOM_ITERS}")
print(f"Refine random iters: {REFINE_RANDOM_ITERS}")
print(f"Bootstrap finalists: {FINALISTS_FOR_BOOTSTRAP} x {BOOTSTRAP_N}")
print("")

print("Stage 1/3 :: coarse random search")
for i in range(COARSE_RANDOM_ITERS):
    p = sample_random_params()
    k = key_of(p)
    if k in seen:
        continue
    seen.add(k)
    results.append(evaluate_params(p))

    if (i + 1) % PROGRESS_EVERY == 0:
        cur = pd.DataFrame(results)
        print(f"  coarse {i+1}/{COARSE_RANDOM_ITERS} | elapsed {(time.time()-start)/60:.1f} min | best {cur['objective'].max():.6f}")

seed_df = pd.DataFrame(results).sort_values(
    ["objective", "concern_precision", "concern_specificity", "exact_match_rate"],
    ascending=False
).reset_index(drop=True)
seed_params = seed_df.head(150).to_dict("records")

print("\nStage 2/3 :: refinement")
for i in range(REFINE_RANDOM_ITERS):
    if seed_params and random.random() < 0.75:
        p = jitter_params(random.choice(seed_params[:60]))
    else:
        p = sample_random_params()

    k = key_of(p)
    if k in seen:
        continue
    seen.add(k)
    results.append(evaluate_params(p))

    if (i + 1) % PROGRESS_EVERY == 0:
        cur = pd.DataFrame(results)
        print(f"  refine {i+1}/{REFINE_RANDOM_ITERS} | elapsed {(time.time()-start)/60:.1f} min | best {cur['objective'].max():.6f}")

print("\nStage 3/3 :: bootstrap robustness")
res_df = pd.DataFrame(results).drop_duplicates().sort_values(
    ["objective", "concern_precision", "concern_specificity", "exact_match_rate"],
    ascending=False
).reset_index(drop=True)

finalists = res_df.head(FINALISTS_FOR_BOOTSTRAP).copy()
boot_rows = []

param_names = [
    "min_evidence", "extreme_eps", "amb_watch", "vis_watch", "amb_red_cap",
    "vis_red_cap", "amb_yellow_cap", "vis_yellow_cap", "lowrisk_red_score",
    "lowrisk_yel_score", "ambig_extra_margin", "moderate_vis_cap"
]

for i, row in finalists.iterrows():
    params = {k: row[k] for k in param_names}
    boot = bootstrap_eval(params, n_boot=BOOTSTRAP_N)
    out = row.to_dict()
    out.update(boot)
    out["robust_score"] = boot["bootstrap_obj_mean"] - 0.50 * boot["bootstrap_obj_std"]
    boot_rows.append(out)

    if (i + 1) % 10 == 0:
        print(f"  bootstrap {i+1}/{len(finalists)} | elapsed {(time.time()-start)/60:.1f} min")

boot_df = pd.DataFrame(boot_rows).sort_values(
    ["robust_score", "bootstrap_precision_mean", "bootstrap_specificity_mean", "bootstrap_exact_mean"],
    ascending=False
).reset_index(drop=True)

best = boot_df.iloc[0].to_dict()
best_params = {k: best[k] for k in param_names}
best_pred_rank = apply_params_to_rank(best_params)
best_metrics = metrics_from_pred(best_pred_rank, ref_rank)

# =========================================================
# SAVE
# =========================================================

out_feed = df.copy()
out_feed["severity_sweep_best"] = [RANK2SEV[int(x)] for x in best_pred_rank]
out_feed["match_vs_ref"] = (out_feed["severity_sweep_best"] == out_feed["validated_ref_severity"]).astype(int)

def action_from_sev(sev):
    sev = str(sev).upper()
    if sev == "RED":
        return "inspect_immediately"
    if sev == "YELLOW":
        return "review_context_and_monitor"
    if sev == "WATCH":
        return "collect_more_visual_evidence"
    return "standard_monitoring"

out_feed["recommended_action_sweep_best"] = out_feed["severity_sweep_best"].apply(action_from_sev)

leaderboard_out = OUT_DIR / "cvb_param_sweep_leaderboard_v2.csv"
bootstrap_out = OUT_DIR / "cvb_param_sweep_bootstrap_v2.csv"
best_feed_out = OUT_DIR / "cvb_param_sweep_best_feed_v2.csv"
best_params_out = OUT_DIR / "cvb_param_sweep_best_params_v2.json"
summary_out = OUT_DIR / "cvb_param_sweep_summary_v2.txt"

res_df.head(10000).to_csv(leaderboard_out, index=False)
boot_df.to_csv(bootstrap_out, index=False)
out_feed.to_csv(best_feed_out, index=False)

with open(best_params_out, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

lines = []
lines.append("CVB PARAM SWEEP V2 SUMMARY")
lines.append("==========================")
lines.append(f"rows: {len(df)}")
lines.append(f"elapsed_minutes: {(time.time()-start)/60:.2f}")
lines.append("")
lines.append("Baseline V4 metrics:")
for k, v in baseline_v4_metrics.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("Best params:")
for k, v in best_params.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("Best metrics:")
for k, v in best_metrics.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append(f"best_objective: {best['objective']}")
lines.append(f"best_robust_score: {best['robust_score']}")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB PARAM SWEEP V2 :: DONE ===")
print(f"Elapsed minutes: {(time.time()-start)/60:.2f}")

print("\nBaseline V4 metrics:")
for k, v in baseline_v4_metrics.items():
    print(f"{k}: {v}")

print("\nBest params:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nBest metrics:")
for k, v in best_metrics.items():
    print(f"{k}: {v}")

print(f"\nSaved:")
print(f"- {leaderboard_out}")
print(f"- {bootstrap_out}")
print(f"- {best_feed_out}")
print(f"- {best_params_out}")
print(f"- {summary_out}")

