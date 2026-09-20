"""Evaluation helpers for simulated AERIS unit states."""

from __future__ import annotations

import pandas as pd

SEVERITY_RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}


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


def binary_metrics(predicted: pd.Series, truth: pd.Series) -> dict[str, float | int]:
    pred = predicted.astype(str).str.upper().map(SEVERITY_RANK).fillna(0).astype(int)
    actual = truth.astype(str).str.upper().map(SEVERITY_RANK).fillna(0).astype(int)

    pred_concern = pred >= 2
    truth_concern = actual >= 2

    tp = int((pred_concern & truth_concern).sum())
    fp = int((pred_concern & ~truth_concern).sum())
    fn = int((~pred_concern & truth_concern).sum())
    tn = int((~pred_concern & ~truth_concern).sum())

    return {
        "exact_match_rate": float((pred == actual).mean()),
        "precision": tp / (tp + fp + 1e-12),
        "recall": tp / (tp + fn + 1e-12),
        "specificity": tn / (tn + fp + 1e-12),
        "accuracy": (tp + tn) / max(len(pred), 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "red_rate": float((pred == 3).mean()),
        "concern_rate": float((pred >= 2).mean()),
    }


def evaluate_units(
    units: pd.DataFrame,
    *,
    total_hours: int,
) -> pd.DataFrame:
    rows = []

    for scenario, frame in units.groupby("scenario", sort=True):
        truth = pd.Series(
            [
                scenario_truth(scenario, int(hour), total_hours)
                for hour in frame["hour"]
            ],
            index=frame.index,
        )
        metrics = binary_metrics(frame["unit_severity"], truth)
        rows.append({"scenario": scenario, **metrics})

    return pd.DataFrame(rows)
