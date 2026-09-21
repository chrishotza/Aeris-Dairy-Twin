import re
import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_aeris_projection_v2.csv"
risk_out = exp_dir / "cvb_behavior_code_risk_profile_v1.csv"
clips_out = exp_dir / "cvb_real_alert_candidates_v1.csv"
summary_out = exp_dir / "cvb_real_alert_candidates_summary_v1.txt"

df = pd.read_csv(inp, dtype={"animal_id": str})
df["clip_id"] = df["clip_id"].astype(str)

def extract_beh_code(clip_id: str):
    m = re.search(r'beh(\d+)', clip_id)
    return m.group(1) if m else "unknown"

severity_rank = {"GREEN": 0, "HOLD": 1, "YELLOW": 2, "RED": 3}

df["behavior_code"] = df["clip_id"].apply(extract_beh_code)
df["severity_rank"] = df["aeris_color"].map(severity_rank).fillna(-1)

# ---------------------------------------------------------
# Risk profile by behavior code
# ---------------------------------------------------------
cross = pd.crosstab(df["behavior_code"], df["aeris_color"])

for c in ["GREEN", "HOLD", "YELLOW", "RED"]:
    if c not in cross.columns:
        cross[c] = 0

cross = cross[["GREEN", "HOLD", "YELLOW", "RED"]].copy()
cross["total"] = cross.sum(axis=1)

for c in ["GREEN", "HOLD", "YELLOW", "RED"]:
    cross[f"{c.lower()}_rate"] = cross[c] / cross["total"]

cross["risk_score"] = (
    1.00 * cross["red_rate"] +
    0.60 * cross["yellow_rate"] +
    0.25 * cross["hold_rate"]
)

risk_df = cross.sort_values(["risk_score", "RED", "YELLOW"], ascending=False).reset_index()
risk_df.to_csv(risk_out, index=False)

# ---------------------------------------------------------
# Clip-level candidates
# ---------------------------------------------------------
clip_rows = []
for clip_id, g in df.groupby("clip_id"):
    max_idx = g["severity_rank"].idxmax()
    max_row = g.loc[max_idx]

    clip_rows.append({
        "clip_id": clip_id,
        "behavior_code": max_row["behavior_code"],
        "n_animals_projected": len(g),
        "n_green": int((g["aeris_color"] == "GREEN").sum()),
        "n_hold": int((g["aeris_color"] == "HOLD").sum()),
        "n_yellow": int((g["aeris_color"] == "YELLOW").sum()),
        "n_red": int((g["aeris_color"] == "RED").sum()),
        "max_aeris_color": max_row["aeris_color"],
        "max_aeris_regime": max_row["aeris_regime"],
        "mean_activity_proxy": round(float(g["activity_proxy"].mean()), 4),
        "mean_rest_proxy": round(float(g["rest_proxy"].mean()), 4),
        "mean_anomaly_proxy": round(float(g["anomaly_proxy"].mean()), 4),
        "mean_visibility_proxy": round(float(g["visibility_proxy"].mean()), 4),
        "candidate_score": round(
            1.0 * (g["aeris_color"] == "RED").mean() +
            0.6 * (g["aeris_color"] == "YELLOW").mean() +
            0.25 * (g["aeris_color"] == "HOLD").mean(),
            4
        ),
        "recommended_action": max_row["recommended_action"]
    })

clips_df = pd.DataFrame(clip_rows)
clips_df = clips_df.sort_values(
    ["candidate_score", "n_red", "n_yellow", "n_hold"],
    ascending=False
).reset_index(drop=True)

clips_df.to_csv(clips_out, index=False)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
lines = []
lines.append("CVB REAL ALERT CANDIDATES SUMMARY V1")
lines.append("====================================")
lines.append(f"rows_projected: {len(df)}")
lines.append(f"unique_behavior_codes: {df['behavior_code'].nunique()}")
lines.append(f"unique_clips: {clips_df['clip_id'].nunique()}")
lines.append("")
lines.append("top_behavior_codes_by_risk:")
for _, row in risk_df.head(10).iterrows():
    lines.append(
        f"- beh{row['behavior_code']}: risk_score={row['risk_score']:.4f}, "
        f"RED={row['RED']}, YELLOW={row['YELLOW']}, HOLD={row['HOLD']}, GREEN={row['GREEN']}"
    )
lines.append("")
lines.append("top_clip_candidates:")
for _, row in clips_df.head(20).iterrows():
    lines.append(
        f"- {row['clip_id']} | beh{row['behavior_code']} | "
        f"max={row['max_aeris_color']} | red={row['n_red']} yellow={row['n_yellow']} hold={row['n_hold']} "
        f"| score={row['candidate_score']:.4f} | {row['recommended_action']}"
    )

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB REAL ALERT CANDIDATES V1 ===")
print(f"rows projected        : {len(df)}")
print(f"unique behavior codes : {df['behavior_code'].nunique()}")
print(f"unique clips          : {clips_df['clip_id'].nunique()}")

print("\nTop behavior codes by risk:")
print(risk_df[["behavior_code","GREEN","HOLD","YELLOW","RED","risk_score"]].head(10).to_string(index=False))

print("\nTop clip candidates:")
print(clips_df.head(20).to_string(index=False))

print(f"\nSaved:")
print(f"- {risk_out}")
print(f"- {clips_out}")
print(f"- {summary_out}")

