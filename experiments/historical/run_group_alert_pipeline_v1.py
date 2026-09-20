import pandas as pd
from pathlib import Path

root = Path(".")
inp = root / "07_assets" / "data_placeholders" / "pen_group_states_placeholder_v1.csv"
out = root / "07_assets" / "data_placeholders" / "generated_group_alerts_v1.csv"

df = pd.read_csv(inp, dtype={"group_id": str})
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["group_id", "timestamp"]).reset_index(drop=True)

def probable_cause(row):
    causes = []
    if row["competition_proxy"] > 0.55:
        causes.append("competition_or_crowding")
    if row["heat_stress_index"] > 0.50:
        causes.append("heat_stress")
    if row["group_activity_index"] < 0.50:
        causes.append("group_activity_degradation")
    return ";".join(causes) if causes else "group_transition_pattern"

alerts = []
counter = 1

for group_id, g in df.groupby("group_id"):
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
            confidence = round(0.5 * float(row["validity_score"]) + 0.5 * min(1.0, float(row["animals_red"]) / 6.0), 3)

            alerts.append({
                "alert_id": f"GAUTO_{counter:04d}",
                "timestamp": row["timestamp"].isoformat(),
                "entity_level": "group",
                "entity_id": group_id,
                "severity": color,
                "regime": regime,
                "animals_green": int(row["animals_green"]),
                "animals_yellow": int(row["animals_yellow"]),
                "animals_red": int(row["animals_red"]),
                "probable_cause": probable_cause(row),
                "confidence": confidence,
                "recommended_action": row["recommended_action"],
                "status": "open" if color == "YELLOW" else "acknowledged"
            })
            counter += 1

        prev_color = color

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv(out, index=False)

print("\n=== GROUP ALERT PIPELINE V1 ===")
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
