"""Channel diagnostics reconstructed from the AERIS corpus."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "scenario",
    "run_id",
    "unit_id",
    "hour",
    "animal_score",
    "rumination",
    "activity",
    "locomotion_quality",
    "feeding_engagement",
    "drinking_pressure",
    "respiration_load",
    "thermal_discomfort",
    "management_disruption",
    "visual_anomaly_proxy",
    "ventilation_quality",
    "water_status",
    "feed_delivery_quality",
    "bedding_quality",
}

CHANNELS = (
    "base_head",
    "heat_head",
    "lame_head",
    "feed_head",
    "water_head",
    "vent_head",
    "manage_head",
    "visual_head",
)


def _clamp01(values):
    return np.clip(values, 0.0, 1.0)


def add_channel_heads(df: pd.DataFrame) -> pd.DataFrame:
    """Add the channel-head transformations used by the historical diagnostics."""
    out = df.copy()

    out["heat_head"] = _clamp01(
        0.42 * out["thermal_discomfort"]
        + 0.28 * out["respiration_load"]
        + 0.12 * out["drinking_pressure"]
        + 0.10 * (1.0 - out["activity"])
        + 0.08 * (1.0 - out["ventilation_quality"])
    )
    out["lame_head"] = _clamp01(
        0.48 * (1.0 - out["locomotion_quality"])
        + 0.18 * (1.0 - out["bedding_quality"])
        + 0.16 * out["visual_anomaly_proxy"]
        + 0.10 * (1.0 - out["activity"])
        + 0.08 * (1.0 - out["rumination"])
    )
    out["feed_head"] = _clamp01(
        0.34 * (1.0 - out["feeding_engagement"])
        + 0.28 * (1.0 - out["feed_delivery_quality"])
        + 0.20 * (1.0 - out["rumination"])
        + 0.10 * out["management_disruption"]
        + 0.08 * (1.0 - out["activity"])
    )
    out["water_head"] = _clamp01(
        0.36 * out["drinking_pressure"]
        + 0.28 * (1.0 - out["water_status"])
        + 0.18 * out["thermal_discomfort"]
        + 0.10 * out["respiration_load"]
        + 0.08 * (1.0 - out["activity"])
    )
    out["vent_head"] = _clamp01(
        0.34 * (1.0 - out["ventilation_quality"])
        + 0.30 * out["respiration_load"]
        + 0.20 * out["thermal_discomfort"]
        + 0.08 * out["visual_anomaly_proxy"]
        + 0.08 * (1.0 - out["activity"])
    )
    out["manage_head"] = _clamp01(
        0.46 * out["management_disruption"]
        + 0.18 * (1.0 - out["feed_delivery_quality"])
        + 0.18 * (1.0 - out["water_status"])
        + 0.10 * (1.0 - out["bedding_quality"])
        + 0.08 * (1.0 - out["activity"])
    )
    out["visual_head"] = _clamp01(out["visual_anomaly_proxy"])
    out["base_head"] = _clamp01(out["animal_score"])
    return out


def diagnose_channels(
    df: pd.DataFrame,
    *,
    expected_targets: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run baseline, separability, ranking and target checks."""

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    work = add_channel_heads(df.copy())
    baseline = work[work["scenario"] == "stable_baseline"].copy()
    if baseline.empty:
        raise ValueError("no stable_baseline rows found")

    baseline_rows = []
    stats: dict[str, dict[str, float]] = {}

    for channel in CHANNELS:
        series = baseline[channel]
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        stats[channel] = {
            "mean": mean,
            "std": std if std > 1e-9 else 1e-9,
            "q90": float(series.quantile(0.90)),
            "q95": float(series.quantile(0.95)),
            "q99": float(series.quantile(0.99)),
        }
        baseline_rows.append(
            {
                "channel": channel,
                "baseline_mean": mean,
                "baseline_std": std,
                "baseline_q90": stats[channel]["q90"],
                "baseline_q95": stats[channel]["q95"],
                "baseline_q99": stats[channel]["q99"],
            }
        )

    baseline_df = pd.DataFrame(baseline_rows)

    scenario_rows = []
    for scenario, frame in work.groupby("scenario"):
        for channel in CHANNELS:
            s = stats[channel]
            z = (frame[channel] - s["mean"]) / s["std"]
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "channel": channel,
                    "mean": float(frame[channel].mean()),
                    "std": float(frame[channel].std(ddof=0)),
                    "q90": float(frame[channel].quantile(0.90)),
                    "q95": float(frame[channel].quantile(0.95)),
                    "q99": float(frame[channel].quantile(0.99)),
                    "z_mean_vs_baseline": float(z.mean()),
                    "z_q90_vs_baseline": float(
                        (frame[channel].quantile(0.90) - s["mean"]) / s["std"]
                    ),
                    "z_q95_vs_baseline": float(
                        (frame[channel].quantile(0.95) - s["mean"]) / s["std"]
                    ),
                    "rate_above_baseline_q95": float(
                        (frame[channel] > s["q95"]).mean()
                    ),
                    "rate_above_baseline_q99": float(
                        (frame[channel] > s["q99"]).mean()
                    ),
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)

    rankings = []
    for scenario, frame in scenario_df.groupby("scenario"):
        ordered = frame.sort_values(
            ["z_mean_vs_baseline", "rate_above_baseline_q95", "z_q95_vs_baseline"],
            ascending=False,
        )
        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            rankings.append(
                {
                    "scenario": scenario,
                    "rank": rank,
                    "channel": row.channel,
                    "z_mean_vs_baseline": row.z_mean_vs_baseline,
                    "rate_above_baseline_q95": row.rate_above_baseline_q95,
                    "z_q95_vs_baseline": row.z_q95_vs_baseline,
                }
            )

    ranking_df = pd.DataFrame(rankings)

    targets = expected_targets or {
        "heat_stress_wave": "heat_head",
        "lameness_cluster": "lame_head",
        "feeding_disruption": "feed_head",
        "water_system_issue": "water_head",
        "ventilation_failure": "vent_head",
    }

    checks = []
    for scenario, expected_channel in targets.items():
        match = scenario_df[
            (scenario_df["scenario"] == scenario)
            & (scenario_df["channel"] == expected_channel)
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        status = "good_signal"
        if row["z_mean_vs_baseline"] < 1.0 or row["rate_above_baseline_q95"] < 0.20:
            status = "weak_signal"
        if row["z_mean_vs_baseline"] < 0.3 or row["rate_above_baseline_q95"] < 0.08:
            status = "very_weak_signal"
        checks.append(
            {
                "scenario": scenario,
                "expected_channel": expected_channel,
                "status": status,
                "z_mean_vs_baseline": row["z_mean_vs_baseline"],
                "rate_above_baseline_q95": row["rate_above_baseline_q95"],
                "rate_above_baseline_q99": row["rate_above_baseline_q99"],
            }
        )

    unit_group = (
        work.groupby(["scenario", "run_id", "unit_id", "hour"])[list(CHANNELS)]
        .mean()
        .reset_index()
    )
    unit_rows = []
    for scenario, frame in unit_group.groupby("scenario"):
        for channel in CHANNELS:
            s = stats[channel]
            z = (frame[channel] - s["mean"]) / s["std"]
            unit_rows.append(
                {
                    "scenario": scenario,
                    "channel": channel,
                    "unit_mean": float(frame[channel].mean()),
                    "unit_q90": float(frame[channel].quantile(0.90)),
                    "unit_z_mean_vs_baseline": float(z.mean()),
                    "unit_rate_above_baseline_q95": float(
                        (frame[channel] > s["q95"]).mean()
                    ),
                }
            )

    return {
        "data": work,
        "baseline": baseline_df,
        "scenario": scenario_df,
        "ranking": ranking_df,
        "checks": pd.DataFrame(checks),
        "unit": pd.DataFrame(unit_rows),
    }
