import pandas as pd
import numpy as np
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_behavior_table_v1.csv"
feat_out = exp_dir / "cvb_clip_animal_features_v1.csv"
proj_out = exp_dir / "cvb_aeris_projection_v1.csv"
summary_out = exp_dir / "cvb_aeris_projection_summary_v1.txt"

df = pd.read_csv(inp, dtype={"animal_id": str, "track_id": str})
df["behavior"] = df["behavior"].fillna("missing").astype(str)
df["animal_id"] = df["animal_id"].fillna("unknown").astype(str)
df["source_json"] = df["source_json"].astype(str)

# nos quedamos con anotaciones visibles si se puede, pero sin romper el dataset
if "occluded" in df.columns:
    df["occluded"] = df["occluded"].fillna(False)

# agregación por clip + animal
counts = (
    df.groupby(["source_json", "animal_id", "behavior"])
      .size()
      .reset_index(name="count")
)

pivot = (
    counts.pivot_table(
        index=["source_json", "animal_id"],
        columns="behavior",
        values="count",
        fill_value=0,
        aggfunc="sum"
    )
    .reset_index()
)

# asegurar columnas esperadas
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

behavior_cols = [c for c in expected if c in pivot.columns]
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

# diversidad de comportamiento observado
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

    # no vendemos magia: si casi no se ve, HOLD
    if total_obs < 8 or visibility < 0.30:
        return ("HOLD", "low_visibility", "collect_more_visual_evidence")

    # anomalía observable fuerte
    if running > 0.08 or anomaly > 0.22:
        return ("RED", "behavior_anomaly", "inspect_immediately")

    # inestabilidad o patrón raro pero no extremo
    if anomaly > 0.10 or (active + rest) < 0.65:
        return ("YELLOW", "behavior_shift", "review_context_and_monitor")

    # estable observable
    return ("GREEN", "stable_observable", "standard_monitoring")

proj = pivot.copy()
states = proj.apply(project_state, axis=1)
proj["aeris_color"] = [x[0] for x in states]
proj["aeris_regime"] = [x[1] for x in states]
proj["recommended_action"] = [x[2] for x in states]

proj.to_csv(proj_out, index=False)
pivot.to_csv(feat_out, index=False)

lines = []
lines.append("CVB AERIS PROJECTION SUMMARY V1")
lines.append("===============================")
lines.append(f"rows_projected: {len(proj)}")
lines.append("")
lines.append("color_counts:")
for k, v in proj["aeris_color"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("regime_counts:")
for k, v in proj["aeris_regime"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("top_behavior_means:")
for col in ["activity_proxy", "rest_proxy", "anomaly_proxy", "visibility_proxy"]:
    lines.append(f"- {col}: {proj[col].mean():.4f}")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB -> AERIS PROJECTION V1 ===")
print(f"rows projected : {len(proj)}")

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
        "source_json","animal_id","total_obs","activity_proxy","rest_proxy",
        "anomaly_proxy","visibility_proxy","behavior_diversity",
        "aeris_color","aeris_regime","recommended_action"
    ]]
    .head(20)
    .to_string(index=False)
)

print(f"\nSaved:")
print(f"- {feat_out}")
print(f"- {proj_out}")
print(f"- {summary_out}")

