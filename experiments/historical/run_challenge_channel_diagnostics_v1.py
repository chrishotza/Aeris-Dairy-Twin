import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SIM_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_simulation_v1"
OUT_DIR = ROOT / "11_real_data" / "cvb_data" / "exports" / "challenge_channel_diagnostics_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_IN = SIM_DIR / "animal_states_challenge_sim_v1.csv"

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def safe_read(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

df = safe_read(ANIMAL_IN)

needed = [
    "scenario","hour","run_id","unit_id","group_id","animal_id","animal_score",
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
    if c not in ["scenario","hour","run_id","unit_id","group_id","animal_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# ---------------------------------------------------------
# CHANNEL HEADS (same family as channel sweep)
# ---------------------------------------------------------
df["heat_head"] = clamp01(
    0.42 * df["thermal_discomfort"] +
    0.28 * df["respiration_load"] +
    0.12 * df["drinking_pressure"] +
    0.10 * (1.0 - df["activity"]) +
    0.08 * (1.0 - df["ventilation_quality"])
)

df["lame_head"] = clamp01(
    0.48 * (1.0 - df["locomotion_quality"]) +
    0.18 * (1.0 - df["bedding_quality"]) +
    0.16 * df["visual_anomaly_proxy"] +
    0.10 * (1.0 - df["activity"]) +
    0.08 * (1.0 - df["rumination"])
)

df["feed_head"] = clamp01(
    0.34 * (1.0 - df["feeding_engagement"]) +
    0.28 * (1.0 - df["feed_delivery_quality"]) +
    0.20 * (1.0 - df["rumination"]) +
    0.10 * df["management_disruption"] +
    0.08 * (1.0 - df["activity"])
)

df["water_head"] = clamp01(
    0.36 * df["drinking_pressure"] +
    0.28 * (1.0 - df["water_status"]) +
    0.18 * df["thermal_discomfort"] +
    0.10 * df["respiration_load"] +
    0.08 * (1.0 - df["activity"])
)

df["vent_head"] = clamp01(
    0.34 * (1.0 - df["ventilation_quality"]) +
    0.30 * df["respiration_load"] +
    0.20 * df["thermal_discomfort"] +
    0.08 * df["visual_anomaly_proxy"] +
    0.08 * (1.0 - df["activity"])
)

df["manage_head"] = clamp01(
    0.46 * df["management_disruption"] +
    0.18 * (1.0 - df["feed_delivery_quality"]) +
    0.18 * (1.0 - df["water_status"]) +
    0.10 * (1.0 - df["bedding_quality"]) +
    0.08 * (1.0 - df["activity"])
)

df["visual_head"] = clamp01(df["visual_anomaly_proxy"])
df["base_head"] = clamp01(df["animal_score"])

channel_cols = [
    "base_head","heat_head","lame_head","feed_head",
    "water_head","vent_head","manage_head","visual_head"
]

# ---------------------------------------------------------
# BASELINE REFERENCE
# ---------------------------------------------------------
baseline = df[df["scenario"] == "stable_baseline"].copy()
if baseline.empty:
    raise ValueError("No stable_baseline rows found")

baseline_rows = []
baseline_stats = {}

for c in channel_cols:
    s = baseline[c]
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    q90 = float(s.quantile(0.90))
    q95 = float(s.quantile(0.95))
    q99 = float(s.quantile(0.99))

    baseline_stats[c] = {
        "mean": mean,
        "std": std if std > 1e-9 else 1e-9,
        "q90": q90,
        "q95": q95,
        "q99": q99,
    }

    baseline_rows.append({
        "channel": c,
        "baseline_mean": mean,
        "baseline_std": std,
        "baseline_q90": q90,
        "baseline_q95": q95,
        "baseline_q99": q99,
    })

baseline_df = pd.DataFrame(baseline_rows)

# ---------------------------------------------------------
# SCENARIO SEPARABILITY
# ---------------------------------------------------------
scenario_rows = []

for scen, g in df.groupby("scenario"):
    for c in channel_cols:
        bs = baseline_stats[c]
        z = (g[c] - bs["mean"]) / bs["std"]

        scenario_rows.append({
            "scenario": scen,
            "channel": c,
            "mean": float(g[c].mean()),
            "std": float(g[c].std(ddof=0)),
            "q90": float(g[c].quantile(0.90)),
            "q95": float(g[c].quantile(0.95)),
            "q99": float(g[c].quantile(0.99)),
            "z_mean_vs_baseline": float(z.mean()),
            "z_q90_vs_baseline": float((g[c].quantile(0.90) - bs["mean"]) / bs["std"]),
            "z_q95_vs_baseline": float((g[c].quantile(0.95) - bs["mean"]) / bs["std"]),
            "rate_above_baseline_q95": float((g[c] > bs["q95"]).mean()),
            "rate_above_baseline_q99": float((g[c] > bs["q99"]).mean()),
        })

scenario_df = pd.DataFrame(scenario_rows)

# ---------------------------------------------------------
# TOP CHANNELS PER SCENARIO
# ---------------------------------------------------------
rank_rows = []
for scen, g in scenario_df.groupby("scenario"):
    g = g.sort_values(
        ["z_mean_vs_baseline", "rate_above_baseline_q95", "z_q95_vs_baseline"],
        ascending=False
    ).reset_index(drop=True)
    for rank, (_, row) in enumerate(g.iterrows(), start=1):
        rank_rows.append({
            "scenario": scen,
            "rank": rank,
            "channel": row["channel"],
            "z_mean_vs_baseline": row["z_mean_vs_baseline"],
            "rate_above_baseline_q95": row["rate_above_baseline_q95"],
            "z_q95_vs_baseline": row["z_q95_vs_baseline"],
        })

rank_df = pd.DataFrame(rank_rows)

# ---------------------------------------------------------
# TARGET CHECKS
# ---------------------------------------------------------
def fetch_metric(scenario, channel):
    sub = scenario_df[(scenario_df["scenario"] == scenario) & (scenario_df["channel"] == channel)]
    if len(sub) == 0:
        return None
    return sub.iloc[0].to_dict()

checks = []

for scenario, expected_channel in [
    ("heat_stress_wave", "heat_head"),
    ("lameness_cluster", "lame_head"),
    ("feeding_disruption", "feed_head"),
    ("water_system_issue", "water_head"),
    ("ventilation_failure", "vent_head"),
]:
    m = fetch_metric(scenario, expected_channel)
    if m is None:
        continue

    status = "good_signal"
    if m["z_mean_vs_baseline"] < 1.0 or m["rate_above_baseline_q95"] < 0.20:
        status = "weak_signal"
    if m["z_mean_vs_baseline"] < 0.3 or m["rate_above_baseline_q95"] < 0.08:
        status = "very_weak_signal"

    checks.append({
        "scenario": scenario,
        "expected_channel": expected_channel,
        "status": status,
        "z_mean_vs_baseline": m["z_mean_vs_baseline"],
        "rate_above_baseline_q95": m["rate_above_baseline_q95"],
        "rate_above_baseline_q99": m["rate_above_baseline_q99"],
    })

checks_df = pd.DataFrame(checks)

# ---------------------------------------------------------
# UNIT-LEVEL CHANNEL DIAGNOSTICS
# ---------------------------------------------------------
unit_group = (
    df.groupby(["scenario","run_id","unit_id","hour"])[channel_cols]
      .mean()
      .reset_index()
)

unit_rows = []
for scen, g in unit_group.groupby("scenario"):
    for c in channel_cols:
        bs = baseline_stats[c]
        z = (g[c] - bs["mean"]) / bs["std"]
        unit_rows.append({
            "scenario": scen,
            "channel": c,
            "unit_mean": float(g[c].mean()),
            "unit_q90": float(g[c].quantile(0.90)),
            "unit_z_mean_vs_baseline": float(z.mean()),
            "unit_rate_above_baseline_q95": float((g[c] > bs["q95"]).mean()),
        })

unit_diag_df = pd.DataFrame(unit_rows)

# ---------------------------------------------------------
# SUMMARY TEXT
# ---------------------------------------------------------
summary_lines = []
summary_lines.append("CHALLENGE CHANNEL DIAGNOSTICS V1")
summary_lines.append("================================")
summary_lines.append("")
summary_lines.append("Baseline stats:")
summary_lines.append(baseline_df.to_string(index=False))
summary_lines.append("")
summary_lines.append("Target checks:")
summary_lines.append(checks_df.to_string(index=False))
summary_lines.append("")
summary_lines.append("Top channels by scenario:")
for scen in rank_df["scenario"].drop_duplicates():
    summary_lines.append("")
    summary_lines.append(f"[{scen}]")
    summary_lines.append(
        rank_df[rank_df["scenario"] == scen]
        .head(5)[["rank","channel","z_mean_vs_baseline","rate_above_baseline_q95","z_q95_vs_baseline"]]
        .to_string(index=False)
    )

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
baseline_out = OUT_DIR / "channel_baseline_stats_v1.csv"
scenario_out = OUT_DIR / "channel_scenario_separability_v1.csv"
rank_out = OUT_DIR / "channel_scenario_ranking_v1.csv"
checks_out = OUT_DIR / "channel_target_checks_v1.csv"
unit_out = OUT_DIR / "channel_unit_diagnostics_v1.csv"
summary_out = OUT_DIR / "channel_diagnostics_summary_v1.txt"

baseline_df.to_csv(baseline_out, index=False)
scenario_df.to_csv(scenario_out, index=False)
rank_df.to_csv(rank_out, index=False)
checks_df.to_csv(checks_out, index=False)
unit_diag_df.to_csv(unit_out, index=False)
summary_out.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n=== CHALLENGE CHANNEL DIAGNOSTICS V1 ===")
print("\nTarget checks:")
print(checks_df.to_string(index=False))

print("\nTop channels by scenario (top 3 each):")
for scen in rank_df["scenario"].drop_duplicates():
    print(f"\n[{scen}]")
    print(
        rank_df[rank_df["scenario"] == scen]
        .head(3)[["rank","channel","z_mean_vs_baseline","rate_above_baseline_q95","z_q95_vs_baseline"]]
        .to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {baseline_out}")
print(f"- {scenario_out}")
print(f"- {rank_out}")
print(f"- {checks_out}")
print(f"- {unit_out}")
print(f"- {summary_out}")

