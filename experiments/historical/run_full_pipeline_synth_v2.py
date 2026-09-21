import pandas as pd
from pathlib import Path

root = Path(".")
data_dir = root / "07_assets" / "data_placeholders"

animal_file = data_dir / "animal_states_synth_v2.csv"
group_file  = data_dir / "pen_group_states_synth_v2.csv"
unit_file   = data_dir / "unit_farm_states_synth_v2.csv"

animal_out = data_dir / "generated_animal_alerts_synth_v2.csv"
group_out  = data_dir / "generated_group_alerts_synth_v2.csv"
unit_out   = data_dir / "generated_unit_alerts_synth_v2.csv"
feed_out   = data_dir / "full_alert_feed_synth_v2.csv"
summary_out= data_dir / "full_alert_summary_synth_v2.txt"

# =========================================================
# Animal pipeline
# =========================================================

animal = pd.read_csv(animal_file, dtype={"animal_id": str, "group_id": str})
animal["timestamp"] = pd.to_datetime(animal["timestamp"])
animal = animal.sort_values(["animal_id", "timestamp"]).reset_index(drop=True)

def animal_probable_cause(row):
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

animal_alerts = []
counter = 1
for animal_id, g in animal.groupby("animal_id"):
    g = g.sort_values("timestamp").reset_index(drop=True)
    prev_color = "GREEN"
    for _, row in g.iterrows():
        color = row["welfare_color"]
        regime = row["regime"]

        should_alert = (
            (color == "YELLOW" and prev_color == "GREEN") or
            (color == "RED" and prev_color != "RED")
        )

        if should_alert:
            confidence = round(0.5 * float(row["validity_score"]) + 0.5 * float(row["anomaly_score"]), 3)
            animal_alerts.append({
                "alert_id": f"AUTO_{counter:05d}",
                "timestamp": row["timestamp"].isoformat(),
                "entity_level": "animal",
                "entity_id": animal_id,
                "severity": color,
                "regime": regime,
                "probable_cause": animal_probable_cause(row),
                "confidence": confidence,
                "recommended_action": row["recommended_action"],
                "status": "open" if color == "YELLOW" else "acknowledged",
                "source_level": "animal"
            })
            counter += 1
        prev_color = color

animal_alerts_df = pd.DataFrame(animal_alerts)
animal_alerts_df.to_csv(animal_out, index=False)

# =========================================================
# Group pipeline
# =========================================================

group = pd.read_csv(group_file, dtype={"group_id": str})
group["timestamp"] = pd.to_datetime(group["timestamp"])
group = group.sort_values(["group_id", "timestamp"]).reset_index(drop=True)

def group_probable_cause(row):
    causes = []
    if row["competition_proxy"] > 0.55:
        causes.append("competition_or_crowding")
    if row["heat_stress_index"] > 0.50:
        causes.append("heat_stress")
    if row["group_activity_index"] < 0.50:
        causes.append("group_activity_degradation")
    return ";".join(causes) if causes else "group_transition_pattern"

group_alerts = []
counter = 1
for group_id, g in group.groupby("group_id"):
    g = g.sort_values("timestamp").reset_index(drop=True)
    prev_color = "GREEN"
    for _, row in g.iterrows():
        color = row["welfare_state"]
        regime = row["regime"]

        should_alert = (
            (color == "YELLOW" and prev_color == "GREEN") or
            (color == "RED" and prev_color != "RED")
        )

        if should_alert:
            confidence = round(0.5 * float(row["validity_score"]) + 0.5 * min(1.0, float(row["animals_red"]) / 6.0), 3)
            group_alerts.append({
                "alert_id": f"GAUTO_{counter:05d}",
                "timestamp": row["timestamp"].isoformat(),
                "entity_level": "group",
                "entity_id": group_id,
                "severity": color,
                "regime": regime,
                "probable_cause": group_probable_cause(row),
                "confidence": confidence,
                "recommended_action": row["recommended_action"],
                "status": "open" if color == "YELLOW" else "acknowledged",
                "source_level": "group"
            })
            counter += 1
        prev_color = color

group_alerts_df = pd.DataFrame(group_alerts)
group_alerts_df.to_csv(group_out, index=False)

# =========================================================
# Unit pipeline
# =========================================================

unit = pd.read_csv(unit_file, dtype={"unit_id": str})
unit["timestamp"] = pd.to_datetime(unit["timestamp"])
unit = unit.sort_values(["unit_id", "timestamp"]).reset_index(drop=True)

def unit_probable_cause(row):
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

unit_alerts = []
counter = 1
for unit_id, g in unit.groupby("unit_id"):
    g = g.sort_values("timestamp").reset_index(drop=True)
    prev_color = "GREEN"
    for _, row in g.iterrows():
        color = row["welfare_state"]
        regime = row["regime"]

        should_alert = (
            (color == "YELLOW" and prev_color == "GREEN") or
            (color == "RED" and prev_color != "RED")
        )

        if should_alert:
            confidence = round(0.5 * float(row["validity_score"]) + 0.5 * (1.0 if color == "RED" else 0.6), 3)
            unit_alerts.append({
                "alert_id": f"UAUTO_{counter:05d}",
                "timestamp": row["timestamp"].isoformat(),
                "entity_level": "unit",
                "entity_id": unit_id,
                "severity": color,
                "regime": regime,
                "probable_cause": unit_probable_cause(row),
                "confidence": confidence,
                "recommended_action": row["recommended_action"],
                "status": "open" if color == "YELLOW" else "acknowledged",
                "source_level": "unit"
            })
            counter += 1
        prev_color = color

unit_alerts_df = pd.DataFrame(unit_alerts)
unit_alerts_df.to_csv(unit_out, index=False)

# =========================================================
# Full feed
# =========================================================

feed = pd.concat([animal_alerts_df, group_alerts_df, unit_alerts_df], ignore_index=True)
feed["timestamp"] = pd.to_datetime(feed["timestamp"])
feed = feed.sort_values(["timestamp", "entity_level", "entity_id"]).reset_index(drop=True)
feed.to_csv(feed_out, index=False)

lines = []
lines.append("FULL ALERT SUMMARY SYNTH V2")
lines.append("===========================")
lines.append(f"animal_alerts: {len(animal_alerts_df)}")
lines.append(f"group_alerts: {len(group_alerts_df)}")
lines.append(f"unit_alerts: {len(unit_alerts_df)}")
lines.append(f"total_alerts: {len(feed)}")
lines.append("")
lines.append("alerts_by_severity:")
for k, v in feed["severity"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("alerts_by_level:")
for k, v in feed["entity_level"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== FULL PIPELINE SYNTH V2 ===")
print(f"animal alerts : {len(animal_alerts_df)}")
print(f"group alerts  : {len(group_alerts_df)}")
print(f"unit alerts   : {len(unit_alerts_df)}")
print(f"total alerts  : {len(feed)}")

print("\nalerts by severity:")
print(feed['severity'].value_counts())

print("\nalerts by level:")
print(feed['entity_level'].value_counts())

print(f"\nSaved:")
print(f"- {animal_out}")
print(f"- {group_out}")
print(f"- {unit_out}")
print(f"- {feed_out}")
print(f"- {summary_out}")

