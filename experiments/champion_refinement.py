"""Clean, configurable reconstruction of the AERIS champion search.

This experiment follows the structure of the historical reinforced/
ventilation-refinement procedures: feature reinforcement, animal thresholds,
group/unit aggregation, temporal persistence, scenario metrics and a
random-search + local-refinement loop.

It is intentionally a public reconstruction, not a byte-for-byte replay of
the historical challenge filesystem or a claim that the historical champion
numbers can be reproduced without the original input artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}

REQUIRED = [
    "run_id", "scenario", "hour", "unit_id", "group_id", "animal_id",
    "animal_score", "phase", "rumination", "activity", "locomotion_quality",
    "feeding_engagement", "drinking_pressure", "respiration_load",
    "thermal_discomfort", "management_disruption", "visual_anomaly_proxy",
    "ventilation_quality", "water_status", "feed_delivery_quality",
    "bedding_quality",
]


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 2097
    random_iters: int = 80
    top_seeds: int = 12
    refine_iters: int = 160


DEFAULTS = {
    "heat_amp": 0.35,
    "lame_amp": 0.55,
    "vent_amp": 0.55,
    "base_gain": 0.82,
    "heat_w": 0.08,
    "loco_w": 0.08,
    "resp_w": 0.05,
    "manage_w": 0.06,
    "intake_w": 0.07,
    "water_w": 0.06,
    "visual_w": 0.05,
    "animal_y": 0.42,
    "animal_r": 0.60,
    "group_y_share": 0.20,
    "group_r_share": 0.33,
    "group_y_score": 0.44,
    "group_r_score": 0.66,
    "unit_y_share": 0.15,
    "unit_r_share": 0.25,
    "unit_y_score": 0.43,
    "unit_r_score": 0.65,
    "min_red_duration": 2,
    "min_yellow_duration": 3,
}


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def scenario_truth(scenario: str, hour: int, total_hours: int) -> str:
    if scenario == "stable_baseline":
        return "GREEN"

    start = int(total_hours * 0.35)
    peak = int(total_hours * 0.50)
    end = int(total_hours * 0.78)

    if hour < start:
        return "GREEN"
    if hour < peak:
        return "YELLOW"
    if hour < end:
        return "YELLOW" if scenario == "post_event_recovery" else "RED"
    return "WATCH"


def binary_metrics(
    pred: pd.Series,
    truth: pd.Series,
) -> dict[str, float | int]:
    predicted = pred.astype(str).str.upper().map(RANK).fillna(0).astype(int)
    actual = truth.astype(str).str.upper().map(RANK).fillna(0).astype(int)

    concern_pred = predicted >= 2
    concern_truth = actual >= 2

    tp = int((concern_pred & concern_truth).sum())
    fp = int((concern_pred & ~concern_truth).sum())
    fn = int((~concern_pred & concern_truth).sum())
    tn = int((~concern_pred & ~concern_truth).sum())

    return {
        "exact_match_rate": float((predicted == actual).mean()),
        "precision": tp / (tp + fp + 1e-12),
        "recall": tp / (tp + fn + 1e-12),
        "specificity": tn / (tn + fp + 1e-12),
        "accuracy": (tp + tn) / max(len(predicted), 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "red_rate": float((predicted == 3).mean()),
        "concern_rate": float((predicted >= 2).mean()),
    }


def apply_persistence(
    values: list[str],
    min_red: int,
    min_yellow: int,
) -> list[str]:
    out = list(values)
    i = 0

    while i < len(out):
        current = out[i]
        if current not in {"RED", "YELLOW"}:
            i += 1
            continue

        j = i
        while j < len(out) and out[j] == current:
            j += 1

        required = min_red if current == "RED" else min_yellow
        if j - i < required:
            replacement = "WATCH" if current == "RED" else "GREEN"
            out[i:j] = [replacement] * (j - i)

        i = j

    return out


def normalize(params: dict) -> dict:
    p = dict(params)

    for key, low, high in [
        ("heat_amp", 0.0, 0.90),
        ("lame_amp", 0.0, 1.20),
        ("vent_amp", 0.0, 1.20),
        ("base_gain", 0.45, 1.10),
        ("heat_w", 0.0, 0.18),
        ("loco_w", 0.0, 0.20),
        ("resp_w", 0.0, 0.20),
        ("manage_w", 0.0, 0.20),
        ("intake_w", 0.0, 0.20),
        ("water_w", 0.0, 0.20),
        ("visual_w", 0.0, 0.18),
        ("animal_y", 0.20, 0.65),
        ("animal_r", 0.30, 0.90),
        ("group_y_share", 0.05, 0.50),
        ("group_r_share", 0.10, 0.70),
        ("group_y_score", 0.20, 0.90),
        ("group_r_score", 0.30, 1.00),
        ("unit_y_share", 0.05, 0.50),
        ("unit_r_share", 0.10, 0.70),
        ("unit_y_score", 0.20, 0.90),
        ("unit_r_score", 0.30, 1.00),
    ]:
        p[key] = float(np.clip(p[key], low, high))

    p["min_red_duration"] = int(
        np.clip(round(p["min_red_duration"]), 1, 4)
    )
    p["min_yellow_duration"] = int(
        np.clip(round(p["min_yellow_duration"]), 1, 6)
    )

    if p["animal_r"] <= p["animal_y"]:
        p["animal_r"] = min(0.90, p["animal_y"] + 0.05)

    return p


def sample_params(rng: random.Random) -> dict:
    p = DEFAULTS.copy()
    p.update(
        heat_amp=rng.uniform(0.10, 0.60),
        lame_amp=rng.uniform(0.15, 0.85),
        vent_amp=rng.uniform(0.15, 0.85),
        base_gain=rng.uniform(0.60, 0.90),
        heat_w=rng.uniform(0.02, 0.12),
        loco_w=rng.uniform(0.02, 0.12),
        resp_w=rng.uniform(0.01, 0.12),
        manage_w=rng.uniform(0.02, 0.12),
        intake_w=rng.uniform(0.02, 0.12),
        water_w=rng.uniform(0.02, 0.12),
        visual_w=rng.uniform(0.01, 0.10),
        animal_y=rng.uniform(0.38, 0.50),
        animal_r=rng.uniform(0.52, 0.68),
        min_red_duration=rng.choice([1, 2, 3]),
        min_yellow_duration=rng.choice([2, 3, 4]),
    )
    return normalize(p)


def jitter_params(seed: dict, rng: random.Random) -> dict:
    p = dict(seed)

    for key, amount in {
        "heat_amp": 0.08,
        "lame_amp": 0.10,
        "vent_amp": 0.10,
        "base_gain": 0.06,
        "heat_w": 0.02,
        "loco_w": 0.02,
        "resp_w": 0.02,
        "manage_w": 0.02,
        "intake_w": 0.02,
        "water_w": 0.02,
        "visual_w": 0.02,
        "animal_y": 0.02,
        "animal_r": 0.02,
        "group_y_share": 0.02,
        "group_r_share": 0.03,
        "group_y_score": 0.03,
        "group_r_score": 0.03,
        "unit_y_share": 0.02,
        "unit_r_share": 0.03,
        "unit_y_score": 0.03,
        "unit_r_score": 0.03,
    }.items():
        p[key] += rng.uniform(-amount, amount)

    p["min_red_duration"] += rng.choice([-1, 0, 1])
    p["min_yellow_duration"] += rng.choice([-1, 0, 1])

    return normalize(p)


def _group_and_unit_predictions(
    df: pd.DataFrame,
    boosted: np.ndarray,
    params: dict,
):
    work = df[
        ["run_id", "scenario", "hour", "unit_id", "group_id"]
    ].copy()
    work["score"] = boosted
    work["animal_y"] = boosted >= params["animal_y"]
    work["animal_r"] = boosted >= params["animal_r"]

    group_cols = [
        "run_id", "scenario", "hour", "unit_id", "group_id"
    ]
    group_stats = (
        work.groupby(group_cols, sort=False)
        .agg(
            group_mean=("score", "mean"),
            group_yshare=("animal_y", "mean"),
            group_rshare=("animal_r", "mean"),
        )
        .reset_index()
    )

    group_stats["group_rank"] = np.where(
        (group_stats["group_rshare"] >= params["group_r_share"])
        | (group_stats["group_mean"] >= params["group_r_score"]),
        3,
        np.where(
            (group_stats["group_yshare"] >= params["group_y_share"])
            | (group_stats["group_mean"] >= params["group_y_score"]),
            2,
            0,
        ),
    )

    unit_cols = ["run_id", "scenario", "hour", "unit_id"]
    unit_stats = (
        group_stats.groupby(unit_cols, sort=False)
        .agg(
            unit_mean=("group_mean", "mean"),
            unit_yshare=(
                "group_rank",
                lambda series: float((series >= 2).mean()),
            ),
            unit_rshare=(
                "group_rank",
                lambda series: float((series == 3).mean()),
            ),
        )
        .reset_index()
    )

    unit_stats["unit_rank"] = np.where(
        (unit_stats["unit_rshare"] >= params["unit_r_share"])
        | (unit_stats["unit_mean"] >= params["unit_r_score"]),
        3,
        np.where(
            (unit_stats["unit_yshare"] >= params["unit_y_share"])
            | (unit_stats["unit_mean"] >= params["unit_y_score"]),
            2,
            0,
        ),
    )

    unit_meta = (
        df[unit_cols]
        .drop_duplicates()
        .sort_values(
            ["run_id", "scenario", "unit_id", "hour"]
        )
        .reset_index(drop=True)
    )

    rank_map = {0: "GREEN", 2: "YELLOW", 3: "RED"}
    unit_meta = unit_meta.merge(
        unit_stats[unit_cols + ["unit_rank"]],
        on=unit_cols,
        how="left",
        validate="one_to_one",
    )

    final = (
        unit_meta["unit_rank"]
        .map(rank_map)
        .fillna("GREEN")
        .to_numpy(dtype=object)
    )

    unit_index = pd.factorize(
        pd.MultiIndex.from_frame(
            unit_meta[["run_id", "scenario", "unit_id"]]
        )
    )[0]

    for key in np.unique(unit_index):
        indices = np.flatnonzero(unit_index == key)
        final[indices] = apply_persistence(
            final[indices].tolist(),
            params["min_red_duration"],
            params["min_yellow_duration"],
        )

    return unit_meta.drop(columns=["unit_rank"]), final


def evaluate_params(
    df: pd.DataFrame,
    params: dict,
) -> dict:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    scenario = df["scenario"].astype(str).to_numpy()
    phase = clamp01(df["phase"].to_numpy(float))
    heat = df["thermal_discomfort"].to_numpy(float)
    loco_drop = 1.0 - df["locomotion_quality"].to_numpy(float)
    resp = df["respiration_load"].to_numpy(float)
    manage = df["management_disruption"].to_numpy(float)
    intake_drop = 1.0 - df["feeding_engagement"].to_numpy(float)
    water_drop = 1.0 - df["water_status"].to_numpy(float)
    visual = df["visual_anomaly_proxy"].to_numpy(float)
    base = df["animal_score"].to_numpy(float)

    heat_focus = clamp01(
        0.34 * heat
        + 0.26 * resp
        + 0.14 * df["drinking_pressure"].to_numpy(float)
        + 0.14 * (1.0 - df["activity"].to_numpy(float))
        + 0.12 * (1.0 - df["ventilation_quality"].to_numpy(float))
    )

    lame_focus = clamp01(
        0.40 * loco_drop
        + 0.20 * (1.0 - df["bedding_quality"].to_numpy(float))
        + 0.18 * visual
        + 0.12 * (1.0 - df["activity"].to_numpy(float))
        + 0.10 * (1.0 - df["rumination"].to_numpy(float))
    )

    vent_focus = clamp01(
        0.34 * (1.0 - df["ventilation_quality"].to_numpy(float))
        + 0.26 * resp
        + 0.18 * heat
        + 0.10 * visual
        + 0.12 * (1.0 - df["activity"].to_numpy(float))
    )

    boosted_base = clamp01(
        base
        + params["heat_amp"]
        * (scenario == "heat_stress_wave")
        * phase
        * heat_focus
        + params["lame_amp"]
        * (scenario == "lameness_cluster")
        * phase
        * lame_focus
        + params["vent_amp"]
        * (scenario == "ventilation_failure")
        * phase
        * vent_focus
    )

    boosted = clamp01(
        params["base_gain"] * boosted_base
        + params["heat_w"] * heat
        + params["loco_w"] * loco_drop
        + params["resp_w"] * resp
        + params["manage_w"] * manage
        + params["intake_w"] * intake_drop
        + params["water_w"] * water_drop
        + params["visual_w"] * visual
    )

    unit_meta, unit_pred = _group_and_unit_predictions(
        df,
        boosted,
        params,
    )

    total_hours = int(df["hour"].max()) + 1
    truth = pd.Series(
        [
            scenario_truth(str(scenario_name), int(hour), total_hours)
            for scenario_name, hour
            in zip(unit_meta["scenario"], unit_meta["hour"])
        ]
    )

    overall = binary_metrics(pd.Series(unit_pred), truth)

    rows = []
    unit_frame = unit_meta.copy()
    unit_frame["pred_severity"] = unit_pred

    for scenario_name, frame in unit_frame.groupby(
        "scenario",
        sort=True,
    ):
        target = pd.Series(
            [
                scenario_truth(
                    scenario_name,
                    int(hour),
                    total_hours,
                )
                for hour in frame["hour"]
            ]
        )
        row = {"scenario": scenario_name}
        row.update(
            binary_metrics(
                frame["pred_severity"],
                target,
            )
        )
        rows.append(row)

    scenario_metrics = pd.DataFrame(rows)

    baseline = scenario_metrics.loc[
        scenario_metrics["scenario"] == "stable_baseline"
    ].iloc[0] if "stable_baseline" in set(scenario_metrics["scenario"]) else None

    targets = scenario_metrics.loc[
        scenario_metrics["scenario"].isin(
            [
                "heat_stress_wave",
                "lameness_cluster",
                "feeding_disruption",
                "water_system_issue",
                "ventilation_failure",
                "post_event_recovery",
            ]
        )
    ]

    min_target_recall = (
        float(targets["recall"].min())
        if not targets.empty
        else 0.0
    )
    mean_target_recall = (
        float(targets["recall"].mean())
        if not targets.empty
        else 0.0
    )

    baseline_concern = (
        float(baseline["concern_rate"])
        if baseline is not None
        else 0.0
    )
    baseline_specificity = (
        float(baseline["specificity"])
        if baseline is not None
        else 1.0
    )

    objective = (
        2.4 * mean_target_recall
        + 2.0 * min_target_recall
        + 0.9 * overall["precision"]
        + 0.9 * overall["specificity"]
        + 0.7 * overall["accuracy"]
        - 2.0 * baseline_concern
    )

    viable = (
        baseline_concern <= 0.08
        and baseline_specificity >= 0.92
        and overall["precision"] >= 0.55
        and overall["concern_rate"] <= 0.60
    )

    if not viable:
        objective -= 100.0

    return {
        "params": params,
        "metrics": overall,
        "objective": float(objective),
        "viable": bool(viable),
        "scenario_metrics": scenario_metrics,
    }


def run_search(
    df: pd.DataFrame,
    config: SearchConfig,
) -> dict:
    rng = random.Random(config.seed)
    rows = []
    best = None

    for _ in range(config.random_iters):
        params = sample_params(rng)
        result = evaluate_params(df, params)
        rows.append(
            {
                **params,
                **result["metrics"],
                "objective": result["objective"],
                "viable": result["viable"],
            }
        )

        if best is None or result["objective"] > best["objective"]:
            best = result

    leaderboard = pd.DataFrame(rows).sort_values(
        ["objective", "viable", "recall", "specificity"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    seeds = leaderboard.head(
        max(1, config.top_seeds)
    ).to_dict("records")

    for _ in range(config.refine_iters):
        seed = rng.choice(seeds)
        base = {key: seed[key] for key in DEFAULTS}
        result = evaluate_params(
            df,
            jitter_params(base, rng),
        )

        rows.append(
            {
                **result["params"],
                **result["metrics"],
                "objective": result["objective"],
                "viable": result["viable"],
            }
        )

        if best is None or result["objective"] > best["objective"]:
            best = result

    leaderboard = pd.DataFrame(rows).sort_values(
        ["objective", "viable", "recall", "specificity"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return {
        "best": best,
        "leaderboard": leaderboard,
        "config": config,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/champion_refinement"),
    )
    parser.add_argument("--seed", type=int, default=2097)
    parser.add_argument("--random-iters", type=int, default=80)
    parser.add_argument("--top-seeds", type=int, default=12)
    parser.add_argument("--refine-iters", type=int, default=160)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    config = SearchConfig(
        seed=args.seed,
        random_iters=args.random_iters,
        top_seeds=args.top_seeds,
        refine_iters=args.refine_iters,
    )

    result = run_search(df, config)

    args.output.mkdir(parents=True, exist_ok=True)

    leaderboard_path = args.output / "leaderboard.csv"
    scenario_path = args.output / "scenario_metrics.csv"
    params_path = args.output / "best_params.json"

    result["leaderboard"].to_csv(
        leaderboard_path,
        index=False,
    )
    result["best"]["scenario_metrics"].to_csv(
        scenario_path,
        index=False,
    )
    params_path.write_text(
        json.dumps(
            result["best"]["params"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment": "aeris_champion_refinement_public_reconstruction_v1",
        "search_config": asdict(config),
        "best_objective": result["best"]["objective"],
        "best_metrics": result["best"]["metrics"],
        "input_file": str(args.input),
        "artifacts": {},
    }

    for path in [leaderboard_path, scenario_path, params_path]:
        manifest["artifacts"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    (args.output / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
