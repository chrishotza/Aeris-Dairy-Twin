"""CVB -> AERIS projection reconstructed from the corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_BEHAVIORS = (
    "grazing",
    "resting-lying",
    "resting-standing",
    "ruminating-lying",
    "ruminating-standing",
    "drinking",
    "walking",
    "grooming",
    "running",
    "hidden",
    "other",
    "none",
    "missing",
)


def _clip_id(path_value: str) -> str:
    path = Path(str(path_value))
    return path.parent.name or "unknown_clip"


def project_cvb_behavior_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project CVB behavior observations into AERIS animal/clip features."""

    required = {"behavior", "animal_id", "file_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    work = df.copy()
    work["behavior"] = work["behavior"].fillna("missing").astype(str)
    work["animal_id"] = work["animal_id"].fillna("unknown").astype(str)
    work["file_name"] = work["file_name"].fillna("missing").astype(str)
    work["clip_id"] = work["file_name"].map(_clip_id)

    counts = (
        work.groupby(["clip_id", "animal_id", "behavior"])
        .size()
        .reset_index(name="count")
    )

    pivot = (
        counts.pivot_table(
            index=["clip_id", "animal_id"],
            columns="behavior",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )

    for behavior in EXPECTED_BEHAVIORS:
        if behavior not in pivot.columns:
            pivot[behavior] = 0

    pivot["total_obs"] = pivot[list(EXPECTED_BEHAVIORS)].sum(axis=1)

    for behavior in EXPECTED_BEHAVIORS:
        pcol = f"p_{behavior.replace('-', '_')}"
        pivot[pcol] = (
            pivot[behavior] / pivot["total_obs"].replace(0, float("nan"))
        ).fillna(0.0)

    pivot["activity_proxy"] = (
        pivot["p_grazing"]
        + pivot["p_drinking"]
        + pivot["p_walking"]
        + pivot["p_grooming"]
    )
    pivot["rest_proxy"] = (
        pivot["p_resting_lying"]
        + pivot["p_resting_standing"]
        + pivot["p_ruminating_lying"]
        + pivot["p_ruminating_standing"]
    )
    pivot["anomaly_proxy"] = (
        pivot["p_running"] + pivot["p_other"] + pivot["p_none"]
    )
    pivot["visibility_proxy"] = 1.0 - pivot["p_hidden"]

    ratio_cols = [
        "p_grazing",
        "p_resting_lying",
        "p_resting_standing",
        "p_ruminating_lying",
        "p_ruminating_standing",
        "p_drinking",
        "p_walking",
        "p_grooming",
        "p_running",
    ]
    pivot["behavior_diversity"] = (pivot[ratio_cols] > 0.05).sum(axis=1)

    def project(row: pd.Series) -> tuple[str, str, str]:
        total = float(row["total_obs"])
        visibility = float(row["visibility_proxy"])
        anomaly = float(row["anomaly_proxy"])
        active = float(row["activity_proxy"])
        rest = float(row["rest_proxy"])
        running = float(row["p_running"])

        if total < 8 or visibility < 0.30:
            return "HOLD", "low_visibility", "collect_more_visual_evidence"
        if running > 0.08 or anomaly > 0.22:
            return "RED", "behavior_anomaly", "inspect_immediately"
        if anomaly > 0.10 or (active + rest) < 0.65:
            return "YELLOW", "behavior_shift", "review_context_and_monitor"
        return "GREEN", "stable_observable", "standard_monitoring"

    projected = pivot.copy()
    states = projected.apply(project, axis=1)
    projected["aeris_color"] = [state[0] for state in states]
    projected["aeris_regime"] = [state[1] for state in states]
    projected["recommended_action"] = [state[2] for state in states]

    summary = (
        projected["aeris_color"]
        .value_counts()
        .rename_axis("aeris_color")
        .reset_index(name="count")
    )
    summary["unique_clips"] = projected["clip_id"].nunique()
    summary["unique_animals"] = projected["animal_id"].nunique()

    return projected, summary
