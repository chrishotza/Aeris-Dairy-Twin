import pandas as pd
from pathlib import Path

root = Path(".")
inp = root / "07_assets" / "data_placeholders" / "unit_farm_states_placeholder_v1.csv"
out = root / "07_assets" / "data_placeholders" / "generated_unit_alerts_v1.csv"

df = pd.read_csv(inp, dtype={"unit_id": str})
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["unit_id", "timestamp"]).reset_index(drop=True)

def probable_cause(row):
    causes = []
    if row["THI"] >= 75:
        causes.append("heat_stress")
    if row["ventilation_state"] in ["insufficient", "inadequate"]:
        causes.append("ventilation_issue")
    if row["water_system_status"] in ["under_pressure", "compromised"]:
        causes.append("water_system_issue")
    if row["feeding_system_status"] == "stressed":
        causes.append("feeding_system_stress")
    if row["bedding_status"] in ["degraded", "under_review"]:
        causes.append("bedding_or_hygiene_issue")
    return ";".join(causes) if causes else "unit_transition_pattern"

alerts = []
counter = 1

for unit_id, g in df.groupby("unit_id"):
    g = g.sort_values("timestamp").reset_index(drop=True)
    prev_color = "GREEN"

    for _, row in g.iterrows():
        color = row["welfare_state"]
        regime = row["regime"]

        should_alert = False
        if color == "YELLOW" and prev_color == "GREEN":
            should_alert = True
        elif color == "RED" and prev_color != "RED":
            should_alert = True

        if should_alert:
            confidence = round(0.5 * float(row["validity_score"]) + 0.5 * (1.0 if color == "RED" else 0.6), 3)

            alerts.append({
                "alert_id": f"UAUTO_{counter:04d}",
                "timestamp": row["timestamp"].isoformat(),
                "entity_level": "unit",
                "entity_id": unit_id,
                "severity": color,
                "regime": regime,
                "global_alert_level": row["global_alert_level"],
                "highest_risk_groups": row["highest_risk_groups"],
                "probable_cause": probable_cause(row),
                "confidence": confidence,
                "recommended_action": row["recommended_action"],
                "status": "open" if color == "YELLOW" else "acknowledged"
            })
            counter += 1

        prev_color = color

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv(out, index=False)

print("\n=== UNIT ALERT PIPELINE V1 ===")
print(f"input rows        : {len(df)}")
print(f"generated alerts  : {len(alerts_df)}")

if len(alerts_df) > 0:
    print("\nalerts by severity:")
    print(alerts_df["severity"].value_counts())
    print("\npreview:")
    print(alerts_df.to_string(index=False))
else:
    print("\nNo alerts generated.")

print(f"\nSaved: {out}")
