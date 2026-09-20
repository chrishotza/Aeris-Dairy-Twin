import pandas as pd
from pathlib import Path

root = Path(".")
inp = root / "07_assets" / "data_placeholders" / "animal_states_placeholder_v1.csv"
out = root / "07_assets" / "data_placeholders" / "generated_animal_alerts_v2.csv"

df = pd.read_csv(inp, dtype={"animal_id": str})
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["animal_id", "timestamp"]).reset_index(drop=True)

def probable_cause(row):
    causes = []
    if row["locomotion_score"] < 0.60:
        causes.append("locomotion_degradation")
    if row["rumination_proxy"] < 0.60:
        causes.append("rumination_drop")
    if row["thermal_stress_index"] > 0.50:
        causes.append("heat_stress")
    if row["activity_index"] < 0.50:
        causes.append("activity_drop")
    return ";".join(causes) if causes else "early_transition_pattern"

alerts = []
counter = 1

for animal_id, g in df.groupby("animal_id"):
    g = g.sort_values("timestamp").reset_index(drop=True)
    prev_color = "GREEN"

    for _, row in g.iterrows():
        color = row["welfare_color"]
        regime = row["regime"]

        should_alert = False
        if color == "YELLOW" and prev_color == "GREEN":
            should_alert = True
        elif color == "RED" and prev_color != "RED":
            should_alert = True

        if should_alert:
            confidence = round(0.5 * float(row["validity_score"]) + 0.5 * float(row["anomaly_score"]), 3)
            alerts.append({
                "alert_id": f"AUTO_{counter:04d}",
                "timestamp": row["timestamp"].isoformat(),
                "entity_level": "animal",
                "entity_id": animal_id,
                "severity": color,
                "regime": regime,
                "probable_cause": probable_cause(row),
                "confidence": confidence,
                "recommended_action": row["recommended_action"],
                "status": "open" if color == "YELLOW" else "acknowledged"
            })
            counter += 1

        prev_color = color

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv(out, index=False)

print("\n=== ANIMAL ALERT PIPELINE V2 ===")
print(f"input rows        : {len(df)}")
print(f"generated alerts  : {len(alerts_df)}")
print("\npreview:")
print(alerts_df.to_string(index=False))
print(f"\nSaved: {out}")
