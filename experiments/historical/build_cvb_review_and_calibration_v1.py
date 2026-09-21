import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

alerts_in = exp_dir / "cvb_clip_alert_feed_v1.csv"
risk_in   = exp_dir / "cvb_behavior_code_risk_profile_v1.csv"

review_out = exp_dir / "cvb_manual_review_shortlist_v1.csv"
calib_out  = exp_dir / "cvb_behavior_code_calibration_v1.csv"
summary_out= exp_dir / "cvb_manual_review_and_calibration_summary_v1.txt"

alerts = pd.read_csv(alerts_in)
risk   = pd.read_csv(risk_in)

# ---------------------------------------------------------
# Manual review shortlist
# ---------------------------------------------------------
# keep strongest RED first, then YELLOW
severity_rank = {"RED": 0, "YELLOW": 1, "WATCH": 2, "GREEN": 3}
alerts["severity_rank"] = alerts["severity"].map(severity_rank)

review = alerts.sort_values(
    ["severity_rank", "candidate_score", "n_red", "n_yellow", "n_hold"],
    ascending=[True, False, False, False, False]
).copy()

review["review_priority"] = [
    "P1" if s == "RED" else "P2" if s == "YELLOW" else "P3"
    for s in review["severity"]
]

review["review_goal"] = review["severity"].map({
    "RED": "verify strong anomaly / confirm if clip truly looks abnormal",
    "YELLOW": "verify borderline behavior shift / context dependence",
    "WATCH": "verify low-confidence signal or visibility issue",
    "GREEN": "reference normal clip"
})

# cap shortlist
shortlist = pd.concat([
    review[review["severity"] == "RED"].head(20),
    review[review["severity"] == "YELLOW"].head(15),
    review[review["severity"] == "WATCH"].head(10),
]).reset_index(drop=True)

shortlist.to_csv(review_out, index=False)

# ---------------------------------------------------------
# Behavior-code calibration suggestions
# ---------------------------------------------------------
risk = risk.copy()

def calibration_label(row):
    rs = float(row["risk_score"])
    red = int(row["RED"])
    yellow = int(row["YELLOW"])
    hold = int(row["HOLD"])
    green = int(row["GREEN"])

    if rs >= 0.25 or red >= 15:
        return "high_risk_code"
    elif rs >= 0.12 or (red + yellow) >= 40:
        return "moderate_risk_code"
    elif hold > green * 0.20:
        return "visibility_or_ambiguity_code"
    else:
        return "low_risk_code"

risk["calibration_class"] = risk.apply(calibration_label, axis=1)

risk["suggested_adjustment"] = risk["calibration_class"].map({
    "high_risk_code": "tighten mapping toward anomaly-sensitive projection",
    "moderate_risk_code": "keep current mapping but inspect top clips manually",
    "visibility_or_ambiguity_code": "treat with visibility caution / avoid over-alerting",
    "low_risk_code": "safe baseline / likely stable observable"
})

risk.to_csv(calib_out, index=False)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
lines = []
lines.append("CVB REVIEW + CALIBRATION SUMMARY V1")
lines.append("===================================")
lines.append(f"total_alert_rows: {len(alerts)}")
lines.append(f"shortlist_rows: {len(shortlist)}")
lines.append("")
lines.append("shortlist_counts_by_severity:")
for k, v in shortlist["severity"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("behavior_code_calibration:")
for _, row in risk.sort_values('risk_score', ascending=False).iterrows():
    lines.append(
        f"- beh{row['behavior_code']}: risk_score={row['risk_score']:.4f}, "
        f"class={row['calibration_class']}, adjustment={row['suggested_adjustment']}"
    )

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB REVIEW + CALIBRATION V1 ===")
print(f"total alert rows : {len(alerts)}")
print(f"shortlist rows   : {len(shortlist)}")

print("\nShortlist by severity:")
print(shortlist["severity"].value_counts())

print("\nBehavior code calibration:")
print(
    risk[["behavior_code","risk_score","GREEN","HOLD","YELLOW","RED","calibration_class","suggested_adjustment"]]
    .sort_values("risk_score", ascending=False)
    .to_string(index=False)
)

print(f"\nSaved:")
print(f"- {review_out}")
print(f"- {calib_out}")
print(f"- {summary_out}")

