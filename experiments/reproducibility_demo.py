"""Generate a tiny deterministic AERIS public reproducibility package.

The fixture is synthetic and intentionally independent of third-party data.
It exercises animal -> group -> unit aggregation plus alert generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from aeris.aggregation import aggregate_group, aggregate_unit
from aeris.alerts import build_alert_feed


def _action(severity: str) -> str:
    return {
        "GREEN": "standard_monitoring",
        "YELLOW": "next_routine_inspection",
        "RED": "immediate_targeted_inspection",
    }[severity]


def build_demo_animals() -> pd.DataFrame:
    rows = []
    units = ("Unit_1", "Unit_2")
    states_u1 = ("GREEN", "GREEN", "YELLOW", "RED", "YELLOW", "GREEN")

    value_map = {
        "animal_score": {"GREEN": 0.25, "YELLOW": 0.55, "RED": 0.80},
        "activity": {"GREEN": 0.70, "YELLOW": 0.48, "RED": 0.28},
        "rumination": {"GREEN": 0.78, "YELLOW": 0.56, "RED": 0.34},
        "locomotion_quality": {"GREEN": 0.84, "YELLOW": 0.62, "RED": 0.40},
        "thermal_discomfort": {"GREEN": 0.22, "YELLOW": 0.48, "RED": 0.72},
        "feeding_engagement": {"GREEN": 0.80, "YELLOW": 0.58, "RED": 0.38},
        "drinking_pressure": {"GREEN": 0.30, "YELLOW": 0.48, "RED": 0.68},
        "respiration_load": {"GREEN": 0.22, "YELLOW": 0.48, "RED": 0.70},
        "management_disruption": {"GREEN": 0.18, "YELLOW": 0.30, "RED": 0.52},
        "visual_anomaly_proxy": {"GREEN": 0.10, "YELLOW": 0.30, "RED": 0.58},
        "ventilation_quality": {"GREEN": 0.82, "YELLOW": 0.62, "RED": 0.42},
        "water_status": {"GREEN": 0.92, "YELLOW": 0.68, "RED": 0.46},
        "feed_delivery_quality": {"GREEN": 0.90, "YELLOW": 0.70, "RED": 0.48},
        "bedding_quality": {"GREEN": 0.86, "YELLOW": 0.68, "RED": 0.50},
        "phase": {"GREEN": 0.0, "YELLOW": 0.6, "RED": 1.0},
    }
    regime_map = {
        "GREEN": "stable",
        "YELLOW": "transition",
        "RED": "collapse-risk",
    }

    for hour in range(6):
        for unit in units:
            severity = states_u1[hour] if unit == "Unit_1" else "GREEN"
            for animal_no in (1, 2):
                row = {
                    "run_id": 0,
                    "scenario": "demo_escalation_recovery",
                    "hour": hour,
                    "unit_id": unit,
                    "group_id": f"{unit}_Group_1",
                    "animal_id": f"{unit}_Group_1_Animal_{animal_no:02d}",
                    "animal_regime": regime_map[severity],
                    "animal_severity": severity,
                    "animal_action": _action(severity),
                }
                row.update({key: values[severity] for key, values in value_map.items()})
                rows.append(row)

    return pd.DataFrame(rows)


def build_demo_package() -> dict[str, pd.DataFrame]:
    animals = build_demo_animals()

    groups = aggregate_group(animals)
    groups["group_action"] = groups["group_severity"].map(
        {
            "GREEN": "standard_monitoring",
            "YELLOW": "inspect_group_context",
            "RED": "activate_group_intervention",
        }
    )

    units = aggregate_unit(groups)
    units["unit_action"] = units["unit_severity"].map(
        {
            "GREEN": "standard_monitoring",
            "YELLOW": "review_unit_context",
            "RED": "activate_unit_response",
        }
    )

    animal_alerts = build_alert_feed(animals, persistence=True)
    group_alerts = build_alert_feed(groups, persistence=True)
    unit_alerts = build_alert_feed(units, persistence=True)

    alerts = pd.concat(
        [animal_alerts, group_alerts, unit_alerts],
        ignore_index=True,
    )
    if not alerts.empty:
        alerts = alerts.sort_values(
            ["run_id", "hour", "entity_level", "entity_id"]
        ).reset_index(drop=True)

    return {
        "animal_states": animals,
        "group_states": groups,
        "unit_states": units,
        "alerts": alerts,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_package(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)

    tables = build_demo_package()
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    files = [output / f"{name}.csv" for name in sorted(tables)]
    manifest = {
        "schema": "aeris.synthetic.reproducibility.v1",
        "synthetic": True,
        "source_commit": _source_commit(),
        "files": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": int(pd.read_csv(path).shape[0]),
            }
            for path in files
        },
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "fixture": "AERIS deterministic demo",
        "source_commit": manifest["source_commit"],
        "entities": {
            "animals": len(tables["animal_states"]),
            "groups": len(tables["group_states"]),
            "units": len(tables["unit_states"]),
            "alerts": len(tables["alerts"]),
        },
        "expected_unit1_path": [
            "GREEN",
            "GREEN",
            "YELLOW",
            "RED",
            "YELLOW",
            "GREEN",
        ],
        "verification": "rebuild fixture and compare CSV SHA256 values",
    }

    (output / "reproducibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/demo"),
    )
    args = parser.parse_args()

    manifest = write_package(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
