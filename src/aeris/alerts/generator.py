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

    if (type(min_red_duration) is not int or type(min_yellow_duration) is not int
            or min_red_duration < 1 or min_yellow_duration < 1):
        raise ValueError("durations must be positive integers")

    out = list(sequence)
    if any(state not in RANK for state in out):
        raise ValueError("unknown severity: expected GREEN, WATCH, YELLOW or RED")
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
    # Grouping drops NaN keys by default; never silently omit a run, scenario
    # or timestamp from the resulting alert feed.
    if states[["run_id", "scenario", "hour"]].isna().any().any():
        raise ValueError("alert stream keys and hours must not be missing")
    hours = pd.to_numeric(states["hour"], errors="coerce")
    if (hours.isna().any() or not hours.map(
            lambda value: bool(pd.notna(value)) and float(value).is_integer()
        ).all() or any(isinstance(value, bool) for value in states["hour"])):
        raise ValueError("hours must be finite whole numbers")

    outputs: list[dict[str, object]] = []

    definitions = (
        ("animal", "animal_id", "animal_severity", "animal_regime", "animal_action"),
        ("group", "group_id", "group_severity", "group_regime", "group_action"),
        ("unit", "unit_id", "unit_severity", "unit_regime", "unit_action"),
    )

    for level, entity_id, severity_col, regime_col, action_col in definitions:
        if not {entity_id, severity_col, regime_col, action_col}.issubset(states.columns):
            continue
        if states[entity_id].isna().any():
            raise ValueError(f"{level} alert entity ID must not be missing")
        if not states[severity_col].isin(RANK).all():
            raise ValueError(f"{level} alert has unknown severity")

        # Entity IDs can be reused between independent simulations. Persist and
        # detect escalations only within the same run and scenario.
        stream_keys = ["run_id", "scenario", entity_id]
        subset = states.sort_values(stream_keys + ["hour"], kind="stable").copy()

        for (_, _, entity_value), group in subset.groupby(stream_keys, sort=False):
            sequence = group[severity_col].astype(str).tolist()
            if persistence:
                sequence = apply_persistence(
                    sequence,
                    min_red_duration=min_red_duration,
                    min_yellow_duration=min_yellow_duration,
                )

            previous = "GREEN"
            for (_, row), current in zip(group.iterrows(), sequence):
                if RANK[current] > RANK[previous]:
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
        ["run_id", "scenario", "entity_level", "entity_id", "hour"]
    ).reset_index(drop=True)
    result.insert(
        0,
        "alert_id",
        [f"SIMALT_{i + 1:06d}" for i in range(len(result))],
    )
    return result
