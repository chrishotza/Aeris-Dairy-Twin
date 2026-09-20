"""Compact reproduction of the documented AERIS synthetic validation logic.

This file intentionally mirrors the mathematical structure described in the
source corpus rather than importing the historical scripts verbatim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEED = 123


def generate_series(regime: str, n: int = 48, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(SEED)
    t = np.arange(n)

    if regime == "stable":
        activity = 0.80 + 0.03 * np.sin(t / 4) + rng.normal(0, 0.02, n)
        rumination = 0.84 + 0.02 * np.sin(t / 5) + rng.normal(0, 0.02, n)
        locomotion = 0.88 + 0.02 * np.sin(t / 6) + rng.normal(0, 0.015, n)
        heat = 0.22 + 0.03 * np.sin(t / 7) + rng.normal(0, 0.02, n)
    elif regime == "transition":
        activity = 0.80 - 0.004 * t + rng.normal(0, 0.025, n)
        rumination = 0.84 - 0.005 * t + rng.normal(0, 0.025, n)
        locomotion = 0.88 - 0.0045 * t + rng.normal(0, 0.02, n)
        heat = 0.22 + 0.004 * t + rng.normal(0, 0.02, n)
    elif regime == "collapse":
        activity = 0.80 - 0.010 * t + rng.normal(0, 0.03, n)
        rumination = 0.84 - 0.011 * t + rng.normal(0, 0.03, n)
        locomotion = 0.88 - 0.010 * t + rng.normal(0, 0.025, n)
        heat = 0.22 + 0.008 * t + rng.normal(0, 0.025, n)
    else:
        raise ValueError(f"unknown regime: {regime}")

    return pd.DataFrame(
        {
            "activity": np.clip(activity, 0, 1),
            "rumination": np.clip(rumination, 0, 1),
            "locomotion": np.clip(locomotion, 0, 1),
            "heat": np.clip(heat, 0, 1),
        }
    )


def structural_validity(df: pd.DataFrame) -> float:
    t = np.arange(len(df))

    slopes = [
        np.polyfit(t, df["activity"].to_numpy(), 1)[0],
        np.polyfit(t, df["rumination"].to_numpy(), 1)[0],
        np.polyfit(t, df["locomotion"].to_numpy(), 1)[0],
        np.polyfit(t, df["heat"].to_numpy(), 1)[0],
    ]

    direction = np.mean(
        [
            slopes[0] < -0.002,
            slopes[1] < -0.002,
            slopes[2] < -0.002,
            slopes[3] > 0.002,
        ]
    )

    mid = len(df) // 2
    first = df.iloc[:mid].mean(numeric_only=True)
    second = df.iloc[mid:].mean(numeric_only=True)

    persistence = np.mean(
        [
            second["activity"] < first["activity"],
            second["rumination"] < first["rumination"],
            second["locomotion"] < first["locomotion"],
            second["heat"] > first["heat"],
        ]
    )

    coherence = np.mean(
        [slopes[0] < 0, slopes[1] < 0, slopes[2] < 0, slopes[3] > 0]
    )

    return float(np.clip(0.40 * direction + 0.35 * persistence + 0.25 * coherence, 0, 1))


def classify_window(df: pd.DataFrame) -> tuple[str, float, float]:
    validity = structural_validity(df)

    severity = np.mean(
        [
            df["activity"].iloc[0] - df["activity"].iloc[-1],
            df["rumination"].iloc[0] - df["rumination"].iloc[-1],
            df["locomotion"].iloc[0] - df["locomotion"].iloc[-1],
            df["heat"].iloc[-1] - df["heat"].iloc[0],
        ]
    )

    minimum_transition_severity = 0.05

    if validity < 0.45 or severity < minimum_transition_severity:
        return "GREEN", validity, severity
    if severity < 0.22:
        return "YELLOW", validity, severity
    return "RED", validity, severity


def run() -> dict[str, float]:
    rng = np.random.default_rng(SEED)

    results = []
    for regime in ("stable", "transition", "collapse"):
        for _ in range(40):
            frame = generate_series(regime, rng=rng)
            predicted, validity, severity = classify_window(frame)
            target = {
                "stable": "GREEN",
                "transition": "YELLOW",
                "collapse": "RED",
            }[regime]
            results.append(
                {
                    "truth": target,
                    "prediction": predicted,
                    "validity": validity,
                    "severity": severity,
                }
            )

    df = pd.DataFrame(results)
    accuracy = float((df["truth"] == df["prediction"]).mean())

    return {"accuracy": accuracy}


if __name__ == "__main__":
    print(run())
