import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

alerts_in = exp_dir / "cvb_clip_alert_feed_v1.csv"
calib_in  = exp_dir / "cvb_behavior_code_calibration_v1.csv"

out_csv   = exp_dir / "cvb_clip_alert_feed_v2_calibrated.csv"
out_txt   = exp_dir / "cvb_clip_alert_feed_v2_calibrated_summary.txt"

alerts = pd.read_csv(alerts_in)
calib  = pd.read_csv(calib_in, dtype={"behavior_code": str})

alerts["behavior_code"] = alerts["behavior_code"].astype(str)
merged = alerts.merge(
    calib[["behavior_code", "calibration_class"]],
    on="behavior_code",
    how="left"
)

severity_order = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
inv_severity = {v: k for k, v in severity_order.items()}

def calibrate(row):
    sev = row["severity"]
    cls = row["calibration_class"]
    score = float(row["candidate_score"])
    n_red = int(row["n_red"])
    n_yellow = int(row["n_yellow"])
    n_hold = int(row["n_hold"])
    vis = float(row["mean_visibility_proxy"])
    anomaly = float(row["mean_anomaly_proxy"])

    rank = severity_order.get(sev, 0)

    if cls == "high_risk_code":
        if sev == "YELLOW" and (score >= 0.18 or n_red >= 1):
            rank = max(rank, severity_order["RED"])
        elif sev == "WATCH" and (score >= 0.12 or n_yellow >= 1):
            rank = max(rank, severity_order["YELLOW"])

    elif cls == "moderate_risk_code":
        if sev == "WATCH" and score >= 0.14 and vis >= 0.65:
            rank = max(rank, severity_order["YELLOW"])

    elif cls == "visibility_or_ambiguity_code":
        if sev == "RED" and (n_hold >= n_red or vis < 0.75):
            rank = min(rank, severity_order["YELLOW"])
        elif sev == "YELLOW" and (n_hold >= n_yellow or vis < 0.65):
            rank = min(rank, severity_order["WATCH"])

    elif cls == "low_risk_code":
        if sev == "RED" and score < 0.38 and anomaly < 0.35:
            rank = min(rank, severity_order["YELLOW"])
        elif sev == "YELLOW" and score < 0.18:
            rank = min(rank, severity_order["WATCH"])

    new_sev = inv_severity[rank]

    if new_sev == "RED":
        action = "inspect_immediately"
        priority = "critical"
    elif new_sev == "YELLOW":
        action = "review_context_and_monitor"
        priority = "high"
    elif new_sev == "WATCH":
        action = "collect_more_visual_evidence"
        priority = "medium"
    else:
        action = "standard_monitoring"
        priority = "low"

    return pd.Series([new_sev, action, priority])

merged[["severity_calibrated", "recommended_action_calibrated", "priority_calibrated"]] = merged.apply(calibrate, axis=1)

changed = merged[merged["severity"] != merged["severity_calibrated"]].copy()

merged.to_csv(out_csv, index=False)

lines = []
lines.append("CVB CLIP ALERT FEED V2 CALIBRATED")
lines.append("=================================")
lines.append(f"rows: {len(merged)}")
lines.append("")
lines.append("original_severity_counts:")
for k, v in merged["severity"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("calibrated_severity_counts:")
for k, v in merged["severity_calibrated"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append(f"changed_rows: {len(changed)}")
lines.append("")
lines.append("top_changed_rows:")
for _, row in changed.head(30).iterrows():
    lines.append(
        f"- {row['clip_id']} | beh{row['behavior_code']} | "
        f"{row['severity']} -> {row['severity_calibrated']} | "
        f"class={row['calibration_class']} | score={row['candidate_score']:.4f}"
    )

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB CLIP ALERT FEED V2 CALIBRATED ===")
print(f"rows         : {len(merged)}")
print(f"changed rows : {len(changed)}")

print("\nOriginal severity counts:")
print(merged["severity"].value_counts())

print("\nCalibrated severity counts:")
print(merged["severity_calibrated"].value_counts())

print("\nTop changed rows:")
if len(changed) == 0:
    print("none")
else:
    print(
        changed[[
            "clip_id","behavior_code","severity","severity_calibrated",
            "calibration_class","candidate_score","n_red","n_yellow","n_hold",
            "mean_visibility_proxy","mean_anomaly_proxy"
        ]]
        .head(30)
        .to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

