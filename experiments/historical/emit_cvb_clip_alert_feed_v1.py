import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_real_alert_candidates_v1.csv"
out_csv = exp_dir / "cvb_clip_alert_feed_v1.csv"
out_txt = exp_dir / "cvb_clip_alert_feed_summary_v1.txt"

df = pd.read_csv(inp)

def assign_alert(row):
    score = float(row["candidate_score"])
    n_red = int(row["n_red"])
    n_yellow = int(row["n_yellow"])
    n_hold = int(row["n_hold"])

    if n_red >= 2 or score >= 0.40:
        return ("RED", "inspect_immediately", "critical")
    elif n_red >= 1 or n_yellow >= 2 or score >= 0.22:
        return ("YELLOW", "review_context_and_monitor", "high")
    elif n_hold >= 2 or score >= 0.10:
        return ("WATCH", "collect_more_visual_evidence", "medium")
    else:
        return ("GREEN", "standard_monitoring", "low")

alerts = []
for i, row in df.iterrows():
    severity, action, priority = assign_alert(row)
    alerts.append({
        "alert_id": f"CVBCLIP_{i+1:05d}",
        "clip_id": row["clip_id"],
        "behavior_code": row["behavior_code"],
        "severity": severity,
        "priority": priority,
        "candidate_score": round(float(row["candidate_score"]), 4),
        "n_animals_projected": int(row["n_animals_projected"]),
        "n_green": int(row["n_green"]),
        "n_hold": int(row["n_hold"]),
        "n_yellow": int(row["n_yellow"]),
        "n_red": int(row["n_red"]),
        "max_aeris_color": row["max_aeris_color"],
        "max_aeris_regime": row["max_aeris_regime"],
        "mean_activity_proxy": round(float(row["mean_activity_proxy"]), 4),
        "mean_rest_proxy": round(float(row["mean_rest_proxy"]), 4),
        "mean_anomaly_proxy": round(float(row["mean_anomaly_proxy"]), 4),
        "mean_visibility_proxy": round(float(row["mean_visibility_proxy"]), 4),
        "recommended_action": action
    })

alerts_df = pd.DataFrame(alerts)
alerts_df = alerts_df.sort_values(
    ["severity", "candidate_score", "n_red", "n_yellow"],
    ascending=[True, False, False, False]
).reset_index(drop=True)

severity_order = {"RED": 0, "YELLOW": 1, "WATCH": 2, "GREEN": 3}
alerts_df["sort_key"] = alerts_df["severity"].map(severity_order)
alerts_df = alerts_df.sort_values(
    ["sort_key", "candidate_score", "n_red", "n_yellow"],
    ascending=[True, False, False, False]
).drop(columns=["sort_key"]).reset_index(drop=True)

alerts_df.to_csv(out_csv, index=False)

lines = []
lines.append("CVB CLIP ALERT FEED SUMMARY V1")
lines.append("==============================")
lines.append(f"total_clips: {len(alerts_df)}")
lines.append("")
lines.append("severity_counts:")
for k, v in alerts_df["severity"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("top_red_clips:")
top_red = alerts_df[alerts_df["severity"] == "RED"].head(20)
if len(top_red) == 0:
    lines.append("- none")
else:
    for _, row in top_red.iterrows():
        lines.append(
            f"- {row['clip_id']} | beh{row['behavior_code']} | "
            f"score={row['candidate_score']:.4f} | red={row['n_red']} yellow={row['n_yellow']} hold={row['n_hold']} | "
            f"{row['recommended_action']}"
        )

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB CLIP ALERT FEED V1 ===")
print(f"total clips : {len(alerts_df)}")

print("\nSeverity counts:")
print(alerts_df["severity"].value_counts())

print("\nTop RED clips:")
print(alerts_df[alerts_df["severity"] == "RED"].head(20).to_string(index=False))

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

