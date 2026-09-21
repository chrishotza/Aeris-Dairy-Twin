import pandas as pd
from pathlib import Path

root = Path(".")
data_dir = root / "07_assets" / "data_placeholders"

animal_file = data_dir / "generated_animal_alerts_v2.csv"
group_file  = data_dir / "generated_group_alerts_v1.csv"
unit_file   = data_dir / "generated_unit_alerts_v1.csv"

out_all     = data_dir / "full_alert_feed_v1.csv"
out_summary = data_dir / "full_alert_summary_v1.txt"

animal = pd.read_csv(animal_file, dtype={"entity_id": str})
group  = pd.read_csv(group_file, dtype={"entity_id": str})
unit   = pd.read_csv(unit_file, dtype={"entity_id": str})

animal["timestamp"] = pd.to_datetime(animal["timestamp"])
group["timestamp"]  = pd.to_datetime(group["timestamp"])
unit["timestamp"]   = pd.to_datetime(unit["timestamp"])

animal["source_level"] = "animal"
group["source_level"]  = "group"
unit["source_level"]   = "unit"

common_cols = [
    "alert_id",
    "timestamp",
    "entity_level",
    "entity_id",
    "severity",
    "regime",
    "probable_cause",
    "confidence",
    "recommended_action",
    "status",
    "source_level",
]

animal2 = animal[common_cols].copy()
group2  = group[common_cols].copy()
unit2   = unit[common_cols].copy()

feed = pd.concat([animal2, group2, unit2], ignore_index=True)
feed = feed.sort_values(["timestamp", "entity_level", "entity_id"]).reset_index(drop=True)

feed.to_csv(out_all, index=False)

severity_counts = feed["severity"].value_counts().to_dict()
level_counts = feed["entity_level"].value_counts().to_dict()

latest_ts = str(feed["timestamp"].max()) if len(feed) else "none"
top_red = feed[feed["severity"] == "RED"][["entity_level", "entity_id", "probable_cause", "recommended_action"]]

lines = []
lines.append("FULL ALERT SUMMARY V1")
lines.append("=====================")
lines.append(f"total_alerts: {len(feed)}")
lines.append(f"latest_timestamp: {latest_ts}")
lines.append("")
lines.append("alerts_by_severity:")
for k, v in severity_counts.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("alerts_by_level:")
for k, v in level_counts.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("top_red_alerts:")
if len(top_red) == 0:
    lines.append("- none")
else:
    for _, row in top_red.iterrows():
        lines.append(
            f"- {row['entity_level']} | {row['entity_id']} | {row['probable_cause']} | {row['recommended_action']}"
        )

out_summary.write_text("\n".join(lines), encoding="utf-8")

print("\n=== FULL PIPELINE V1 ===")
print(f"animal alerts : {len(animal)}")
print(f"group alerts  : {len(group)}")
print(f"unit alerts   : {len(unit)}")
print(f"total alerts  : {len(feed)}")

print("\nalerts by severity:")
print(feed["severity"].value_counts())

print("\nalerts by level:")
print(feed["entity_level"].value_counts())

print("\npreview:")
print(feed.to_string(index=False))

print(f"\nSaved feed   : {out_all}")
print(f"Saved summary: {out_summary}")

