import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

alerts_in = exp_dir / "cvb_clip_alert_feed_v1.csv"
calib_in  = exp_dir / "cvb_behavior_code_calibration_v1.csv"

out_csv   = exp_dir / "cvb_clip_alert_feed_v3_conservative.csv"
out_txt   = exp_dir / "cvb_clip_alert_feed_v3_conservative_summary.txt"

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

def conservative_calibrate(row):
    sev = row["severity"]
    cls = row["calibration_class"]
    score = float(row["candidate_score"])
    n_red = int(row["n_red"])
    n_yellow = int(row["n_yellow"])
    n_hold = int(row["n_hold"])
    vis = float(row["mean_visibility_proxy"])
    anomaly = float(row["mean_anomaly_proxy"])

    rank = severity_order.get(sev, 0)
    review_boost = "none"

    # NEVER escalate by code alone.
    # Only de-escalate when ambiguity/visibility suggests caution.

    if cls == "visibility_or_ambiguity_code":
        review_boost = "manual_review_recommended"

        if sev == "RED" and (n_hold >= n_red or vis < 0.75):
            rank = severity_order["YELLOW"]
        elif sev == "YELLOW" and (n_hold >= n_yellow or vis < 0.65):
            rank = severity_order["WATCH"]

    elif cls == "low_risk_code":
        review_boost = "baseline_code"

        if sev == "RED" and score < 0.38 and anomaly < 0.35:
            rank = severity_order["YELLOW"]
        elif sev == "YELLOW" and score < 0.18:
            rank = severity_order["WATCH"]

    elif cls == "moderate_risk_code":
        review_boost = "review_if_possible"
        # no escalation, no automatic downgrade unless very low visibility
        if sev == "YELLOW" and vis < 0.45:
            rank = severity_order["WATCH"]

    elif cls == "high_risk_code":
        review_boost = "priority_review"
        # keep severity as-is; do not auto-upgrade
        if sev == "RED" and vis < 0.40 and n_hold > n_red:
            rank = severity_order["YELLOW"]

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

    return pd.Series([new_sev, action, priority, review_boost])

merged[[
    "severity_conservative",
    "recommended_action_conservative",
    "priority_conservative",
    "review_priority_boost"
]] = merged.apply(conservative_calibrate, axis=1)

changed = merged[merged["severity"] != merged["severity_conservative"]].copy()

merged.to_csv(out_csv, index=False)

lines = []
lines.append("CVB CLIP ALERT FEED V3 CONSERVATIVE")
lines.append("===================================")
lines.append(f"rows: {len(merged)}")
lines.append("")
lines.append("original_severity_counts:")
for k, v in merged["severity"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("conservative_severity_counts:")
for k, v in merged["severity_conservative"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append(f"changed_rows: {len(changed)}")
lines.append("")
lines.append("top_changed_rows:")
for _, row in changed.head(30).iterrows():
    lines.append(
        f"- {row['clip_id']} | beh{row['behavior_code']} | "
        f"{row['severity']} -> {row['severity_conservative']} | "
        f"class={row['calibration_class']} | score={row['candidate_score']:.4f} | "
        f"review={row['review_priority_boost']}"
    )

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB CLIP ALERT FEED V3 CONSERVATIVE ===")
print(f"rows         : {len(merged)}")
print(f"changed rows : {len(changed)}")

print("\nOriginal severity counts:")
print(merged["severity"].value_counts())

print("\nConservative severity counts:")
print(merged["severity_conservative"].value_counts())

print("\nTop changed rows:")
if len(changed) == 0:
    print("none")
else:
    print(
        changed[[
            "clip_id","behavior_code","severity","severity_conservative",
            "calibration_class","candidate_score","n_red","n_yellow","n_hold",
            "mean_visibility_proxy","mean_anomaly_proxy","review_priority_boost"
        ]]
        .head(30)
        .to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

