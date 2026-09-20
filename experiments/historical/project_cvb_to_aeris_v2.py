import pandas as pd
import numpy as np
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_behavior_table_v1.csv"
feat_out = exp_dir / "cvb_clip_animal_features_v2.csv"
proj_out = exp_dir / "cvb_aeris_projection_v2.csv"
summary_out = exp_dir / "cvb_aeris_projection_summary_v2.txt"

df = pd.read_csv(inp, dtype={"animal_id": str, "track_id": str})
df["behavior"] = df["behavior"].fillna("missing").astype(str)
df["animal_id"] = df["animal_id"].fillna("unknown").astype(str)
df["file_name"] = df["file_name"].fillna("missing").astype(str)

def extract_clip_id(path_str: str) -> str:
    try:
        p = Path(path_str)
        return p.parent.name if p.parent.name else "unknown_clip"
    except Exception:
        return "unknown_clip"

df["clip_id"] = df["file_name"].apply(extract_clip_id)

# agregación por clip + animal
counts = (
    df.groupby(["clip_id", "animal_id", "behavior"])
      .size()
      .reset_index(name="count")
)

pivot = (
    counts.pivot_table(
        index=["clip_id", "animal_id"],
        columns="behavior",
        values="count",
        fill_value=0,
        aggfunc="sum"
    )
    .reset_index()
)

expected = [
    "grazing",
    "resting-lying",
    "resting-standing",
    "ruminating-lying",
    "ruminating-standing",
    "drinking",
    "walking",
    "grooming",
    "running",
    "hidden",
    "other",
    "none",
    "missing",
]
for c in expected:
    if c not in pivot.columns:
        pivot[c] = 0

behavior_cols = expected.copy()
pivot["total_obs"] = pivot[behavior_cols].sum(axis=1)

for c in behavior_cols:
    pivot[f"p_{c.replace('-', '_')}"] = np.where(
        pivot["total_obs"] > 0,
        pivot[c] / pivot["total_obs"],
        0.0
    )

pivot["activity_proxy"] = (
    pivot["p_grazing"] +
    pivot["p_drinking"] +
    pivot["p_walking"] +
    pivot["p_grooming"]
)

pivot["rest_proxy"] = (
    pivot["p_resting_lying"] +
    pivot["p_resting_standing"] +
    pivot["p_ruminating_lying"] +
    pivot["p_ruminating_standing"]
)

pivot["anomaly_proxy"] = (
    pivot["p_running"] +
    pivot["p_other"] +
    pivot["p_none"]
)

pivot["visibility_proxy"] = 1.0 - pivot["p_hidden"]

ratio_cols = [
    "p_grazing","p_resting_lying","p_resting_standing","p_ruminating_lying",
    "p_ruminating_standing","p_drinking","p_walking","p_grooming","p_running"
]
pivot["behavior_diversity"] = (pivot[ratio_cols] > 0.05).sum(axis=1)

def project_state(row):
    total_obs = row["total_obs"]
    visibility = row["visibility_proxy"]
    anomaly = row["anomaly_proxy"]
    active = row["activity_proxy"]
    rest = row["rest_proxy"]
    running = row["p_running"]

    if total_obs < 8 or visibility < 0.30:
        return ("HOLD", "low_visibility", "collect_more_visual_evidence")

    if running > 0.08 or anomaly > 0.22:
        return ("RED", "behavior_anomaly", "inspect_immediately")

    if anomaly > 0.10 or (active + rest) < 0.65:
        return ("YELLOW", "behavior_shift", "review_context_and_monitor")

    return ("GREEN", "stable_observable", "standard_monitoring")

proj = pivot.copy()
states = proj.apply(project_state, axis=1)
proj["aeris_color"] = [x[0] for x in states]
proj["aeris_regime"] = [x[1] for x in states]
proj["recommended_action"] = [x[2] for x in states]

pivot.to_csv(feat_out, index=False)
proj.to_csv(proj_out, index=False)

lines = []
lines.append("CVB AERIS PROJECTION SUMMARY V2")
lines.append("===============================")
lines.append(f"rows_projected: {len(proj)}")
lines.append(f"unique_clips: {proj['clip_id'].nunique()}")
lines.append(f"unique_animals: {proj['animal_id'].nunique()}")
lines.append("")
lines.append("color_counts:")
for k, v in proj["aeris_color"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("regime_counts:")
for k, v in proj["aeris_regime"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB -> AERIS PROJECTION V2 ===")
print(f"rows projected : {len(proj)}")
print(f"unique clips   : {proj['clip_id'].nunique()}")
print(f"unique animals : {proj['animal_id'].nunique()}")

print("\nColor counts:")
print(proj["aeris_color"].value_counts())

print("\nRegime counts:")
print(proj["aeris_regime"].value_counts())

print("\nMean proxies:")
print(
    proj[["activity_proxy","rest_proxy","anomaly_proxy","visibility_proxy","behavior_diversity"]]
    .mean()
    .round(4)
)

print("\nPreview:")
print(
    proj[[
        "clip_id","animal_id","total_obs","activity_proxy","rest_proxy",
        "anomaly_proxy","visibility_proxy","behavior_diversity",
        "aeris_color","aeris_regime","recommended_action"
    ]]
    .head(25)
    .to_string(index=False)
)

print(f"\nSaved:")
print(f"- {feat_out}")
print(f"- {proj_out}")
print(f"- {summary_out}")
