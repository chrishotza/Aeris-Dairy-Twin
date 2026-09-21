import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_review_pack_v1.csv"
out = exp_dir / "cvb_manual_adjudication_template_v1.csv"

df = pd.read_csv(inp)

df["reviewed"] = ""
df["human_label"] = ""
df["human_severity"] = ""
df["visibility_ok"] = ""
df["true_concern"] = ""
df["notes"] = ""

cols = [
    "clip_id",
    "behavior_code",
    "severity_conservative",
    "candidate_score",
    "n_red",
    "n_yellow",
    "n_hold",
    "mean_visibility_proxy",
    "mean_anomaly_proxy",
    "recommended_action_conservative",
    "review_priority_boost",
    "reviewed",
    "human_label",
    "human_severity",
    "visibility_ok",
    "true_concern",
    "notes",
]

df[cols].to_csv(out, index=False)

print("\n=== CVB MANUAL ADJUDICATION TEMPLATE V1 ===")
print(f"rows: {len(df)}")
print(f"Saved: {out}")

