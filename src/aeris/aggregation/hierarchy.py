"""Animal -> group -> unit aggregation helpers."""

from __future__ import annotations

import pandas as pd


def aggregate_group(
    animals: pd.DataFrame,
    *,
    yellow_share: float = 0.20,
    red_share: float = 0.33,
    yellow_score: float = 0.44,
    red_score: float = 0.66,
) -> pd.DataFrame:
    required = {
        "run_id", "scenario", "hour", "unit_id", "group_id",
        "animal_score", "animal_severity",
    }
    missing = required - set(animals.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    rows = []
    group_cols = ["run_id", "scenario", "hour", "unit_id", "group_id"]

    for keys, frame in animals.groupby(group_cols, sort=False):
        run_id, scenario, hour, unit_id, group_id = keys
        mean_score = float(frame["animal_score"].mean())
        red = int((frame["animal_severity"] == "RED").sum())
        yellow = int(frame["animal_severity"].isin(["YELLOW", "RED"]).sum())
        total = len(frame)
        red_fraction = red / total
        yellow_fraction = yellow / total

        if red_fraction >= red_share or mean_score >= red_score:
            regime, severity = "collapse-risk", "RED"
        elif yellow_fraction >= yellow_share or mean_score >= yellow_score:
            regime, severity = "transition", "YELLOW"
        else:
            regime, severity = "stable", "GREEN"

        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "hour": hour,
                "unit_id": unit_id,
                "group_id": group_id,
                "group_score": mean_score,
                "group_regime": regime,
                "group_severity": severity,
                "animals_total": total,
                "animals_red": red,
                "animals_yellow": int((frame["animal_severity"] == "YELLOW").sum()),
                "animals_green": int((frame["animal_severity"] == "GREEN").sum()),
            }
        )

    return pd.DataFrame(rows)


def aggregate_unit(
    groups: pd.DataFrame,
    *,
    yellow_share: float = 0.15,
    red_share: float = 0.25,
    yellow_score: float = 0.43,
    red_score: float = 0.65,
) -> pd.DataFrame:
    required = {
        "run_id", "scenario", "hour", "unit_id",
        "group_score", "group_severity",
    }
    missing = required - set(groups.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    rows = []
    for keys, frame in groups.groupby(
        ["run_id", "scenario", "hour", "unit_id"], sort=False
    ):
        run_id, scenario, hour, unit_id = keys
        mean_score = float(frame["group_score"].mean())
        red = int((frame["group_severity"] == "RED").sum())
        yellow = int(frame["group_severity"].isin(["YELLOW", "RED"]).sum())
        total = len(frame)
        red_fraction = red / total
        yellow_fraction = yellow / total

        if red_fraction >= red_share or mean_score >= red_score:
            regime, severity = "collapse-risk", "RED"
        elif yellow_fraction >= yellow_share or mean_score >= yellow_score:
            regime, severity = "transition", "YELLOW"
        else:
            regime, severity = "stable", "GREEN"

        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "hour": hour,
                "unit_id": unit_id,
                "unit_score": mean_score,
                "unit_regime": regime,
                "unit_severity": severity,
                "groups_total": total,
                "groups_red": red,
                "groups_yellow": int((frame["group_severity"] == "YELLOW").sum()),
                "groups_green": int((frame["group_severity"] == "GREEN").sum()),
            }
        )

    return pd.DataFrame(rows)
