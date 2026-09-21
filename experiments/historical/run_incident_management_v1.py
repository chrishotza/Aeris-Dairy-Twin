import pandas as pd
from pathlib import Path

root = Path(".")
data_dir = root / "07_assets" / "data_placeholders"

inp = data_dir / "full_alert_feed_v1.csv"
out_events = data_dir / "managed_alert_events_v1.csv"
out_incidents = data_dir / "incident_summary_v1.csv"

df = pd.read_csv(inp, dtype={"entity_id": str})
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["entity_level", "entity_id", "timestamp"]).reset_index(drop=True)

severity_rank = {"YELLOW": 1, "RED": 2}
cooldown_hours = 3

managed_rows = []
incident_rows = []
incident_counter = 1

for (entity_level, entity_id), g in df.groupby(["entity_level", "entity_id"], sort=False):
    g = g.sort_values("timestamp").reset_index(drop=True)

    active_incident_id = None
    active_opened_at = None
    active_last_event_at = None
    active_max_severity = None
    active_status = None
    active_causes = set()
    active_actions = set()
    active_source_levels = set()

    for _, row in g.iterrows():
        ts = row["timestamp"]
        sev = row["severity"]
        sev_rank = severity_rank.get(sev, 0)
        cause = str(row["probable_cause"])
        action = str(row["recommended_action"])
        src = str(row["source_level"])

        if active_incident_id is None:
            active_incident_id = f"INC_{incident_counter:04d}"
            incident_counter += 1
            active_opened_at = ts
            active_last_event_at = ts
            active_max_severity = sev
            active_status = "open" if sev == "YELLOW" else "acknowledged"
            active_causes = {cause}
            active_actions = {action}
            active_source_levels = {src}

            managed_rows.append({
                "incident_id": active_incident_id,
                "event_type": "opened",
                "timestamp": ts,
                "entity_level": entity_level,
                "entity_id": entity_id,
                "severity": sev,
                "regime": row["regime"],
                "probable_cause": cause,
                "confidence": row["confidence"],
                "recommended_action": action,
                "source_level": src
            })
            continue

        hours_since_last = (ts - active_last_event_at).total_seconds() / 3600.0
        current_max_rank = severity_rank.get(active_max_severity, 0)

        # suppress duplicates inside cooldown if same or weaker severity
        if hours_since_last <= cooldown_hours and sev_rank <= current_max_rank:
            active_last_event_at = ts
            active_causes.add(cause)
            active_actions.add(action)
            active_source_levels.add(src)

            managed_rows.append({
                "incident_id": active_incident_id,
                "event_type": "suppressed_duplicate",
                "timestamp": ts,
                "entity_level": entity_level,
                "entity_id": entity_id,
                "severity": sev,
                "regime": row["regime"],
                "probable_cause": cause,
                "confidence": row["confidence"],
                "recommended_action": action,
                "source_level": src
            })
            continue

        # escalate inside same incident
        if sev_rank > current_max_rank:
            active_last_event_at = ts
            active_max_severity = sev
            active_status = "acknowledged"
            active_causes.add(cause)
            active_actions.add(action)
            active_source_levels.add(src)

            managed_rows.append({
                "incident_id": active_incident_id,
                "event_type": "escalated",
                "timestamp": ts,
                "entity_level": entity_level,
                "entity_id": entity_id,
                "severity": sev,
                "regime": row["regime"],
                "probable_cause": cause,
                "confidence": row["confidence"],
                "recommended_action": action,
                "source_level": src
            })
            continue

        # if cooldown passed, close previous incident and open a new one
        incident_rows.append({
            "incident_id": active_incident_id,
            "entity_level": entity_level,
            "entity_id": entity_id,
            "opened_at": active_opened_at,
            "last_event_at": active_last_event_at,
            "max_severity": active_max_severity,
            "status": active_status,
            "probable_causes": ";".join(sorted(active_causes)),
            "recommended_actions": ";".join(sorted(active_actions)),
            "source_levels": ";".join(sorted(active_source_levels))
        })

        active_incident_id = f"INC_{incident_counter:04d}"
        incident_counter += 1
        active_opened_at = ts
        active_last_event_at = ts
        active_max_severity = sev
        active_status = "open" if sev == "YELLOW" else "acknowledged"
        active_causes = {cause}
        active_actions = {action}
        active_source_levels = {src}

        managed_rows.append({
            "incident_id": active_incident_id,
            "event_type": "opened_new_after_cooldown",
            "timestamp": ts,
            "entity_level": entity_level,
            "entity_id": entity_id,
            "severity": sev,
            "regime": row["regime"],
            "probable_cause": cause,
            "confidence": row["confidence"],
            "recommended_action": action,
            "source_level": src
        })

    # flush final active incident
    if active_incident_id is not None:
        incident_rows.append({
            "incident_id": active_incident_id,
            "entity_level": entity_level,
            "entity_id": entity_id,
            "opened_at": active_opened_at,
            "last_event_at": active_last_event_at,
            "max_severity": active_max_severity,
            "status": active_status,
            "probable_causes": ";".join(sorted(active_causes)),
            "recommended_actions": ";".join(sorted(active_actions)),
            "source_levels": ";".join(sorted(active_source_levels))
        })

events_df = pd.DataFrame(managed_rows).sort_values(["timestamp", "entity_level", "entity_id"]).reset_index(drop=True)
incidents_df = pd.DataFrame(incident_rows).sort_values(["opened_at", "entity_level", "entity_id"]).reset_index(drop=True)

events_df.to_csv(out_events, index=False)
incidents_df.to_csv(out_incidents, index=False)

print("\n=== INCIDENT MANAGEMENT V1 ===")
print(f"raw alerts            : {len(df)}")
print(f"managed events        : {len(events_df)}")
print(f"incidents             : {len(incidents_df)}")

print("\nmanaged event types:")
print(events_df["event_type"].value_counts())

print("\nincident max severity:")
print(incidents_df["max_severity"].value_counts())

print("\nincident preview:")
print(incidents_df.to_string(index=False))

print(f"\nSaved events   : {out_events}")
print(f"Saved incidents: {out_incidents}")

