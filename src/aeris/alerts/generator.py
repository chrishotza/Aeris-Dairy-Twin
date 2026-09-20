"""Alert-generation logic extracted from the historical AERIS simulator."""

from __future__ import annotations

import pandas as pd


RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}


def apply_persistence(
    sequence: list[str],
    *,
    min_red_duration: int = 2,
    min_yellow_duration: int = 2,
) -> list[str]:
    """Suppress short-lived RED/YELLOW runs."""

    if min_red_duration < 1 or min_yellow_duration < 1:
        raise ValueError("durations must be >= 1")

    out = list(sequence)
    i = 0

    while i < len(out):
        current = out[i]
        if current not in {"RED", "YELLOW"}:
            i += 1
            continue

        j = i
        while j < len(out) and out[j] == current:
            j += 1

        required = min_red_duration if current == "RED" else min_yellow_duration
        if j - i < required:
            replacement = "WATCH" if current == "RED" else "GREEN"
            out[i:j] = [replacement] * (j - i)

        i = j

    return out


def build_alert_feed(
    states: pd.DataFrame,
    *,
    persistence: bool = False,
    min_red_duration: int = 2,
    min_yellow_duration: int = 2,
) -> pd.DataFrame:
    """Create escalation events from animal/group/unit state tables.

    Expected columns for each level:
    - <level>_id
    - <level>_severity
    - <level>_regime
    - <level>_action
    - run_id
    - scenario
    - hour
    """

    required = {"run_id", "scenario", "hour"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    outputs: list[dict[str, object]] = []

    definitions = (
        ("animal", "animal_id", "animal_severity", "animal_regime", "animal_action"),
        ("group", "group_id", "group_severity", "group_regime", "group_action"),
        ("unit", "unit_id", "unit_severity", "unit_regime", "unit_action"),
    )

    for level, entity_id, severity_col, regime_col, action_col in definitions:
        if not {entity_id, severity_col, regime_col, action_col}.issubset(states.columns):
            continue

        subset = states.sort_values([entity_id, "hour"]).copy()

        for entity_value, group in subset.groupby(entity_id):
            sequence = group[severity_col].astype(str).tolist()
            if persistence:
                sequence = apply_persistence(
                    sequence,
                    min_red_duration=min_red_duration,
                    min_yellow_duration=min_yellow_duration,
                )

            previous = "GREEN"
            for (_, row), current in zip(group.iterrows(), sequence):
                if RANK.get(current, 0) > RANK.get(previous, 0):
                    outputs.append(
                        {
                            "entity_level": level,
                            "entity_id": entity_value,
                            "hour": int(row["hour"]),
                            "severity": current,
                            "regime": row[regime_col],
                            "recommended_action": row[action_col],
                            "run_id": int(row["run_id"]),
                            "scenario": row["scenario"],
                        }
                    )
                previous = current

    result = pd.DataFrame(outputs)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "entity_level",
                "entity_id",
                "hour",
                "severity",
                "regime",
                "recommended_action",
                "run_id",
                "scenario",
            ]
        )

    result = result.sort_values(
        ["run_id", "entity_level", "entity_id", "hour"]
    ).reset_index(drop=True)
    result.insert(
        0,
        "alert_id",
        [f"SIMALT_{i + 1:06d}" for i in range(len(result))],
    )
    return result
