import numpy as np
import pandas as pd

np.random.seed(2026)

# =========================================================
# Base generator
# =========================================================

def segment(regime: str, n: int):
    t = np.arange(n)

    if regime == "stable":
        activity   = 0.80 + 0.03*np.sin(t/4) + np.random.normal(0, 0.02, n)
        rumination = 0.84 + 0.02*np.sin(t/5) + np.random.normal(0, 0.02, n)
        locomotion = 0.88 + 0.02*np.sin(t/6) + np.random.normal(0, 0.015, n)
        heat       = 0.22 + 0.03*np.sin(t/7) + np.random.normal(0, 0.02, n)

    elif regime == "transition":
        activity   = 0.80 - 0.004*t + np.random.normal(0, 0.025, n)
        rumination = 0.84 - 0.005*t + np.random.normal(0, 0.025, n)
        locomotion = 0.88 - 0.0045*t + np.random.normal(0, 0.02, n)
        heat       = 0.22 + 0.004*t + np.random.normal(0, 0.02, n)

    elif regime == "collapse":
        activity   = 0.80 - 0.010*t + np.random.normal(0, 0.03, n)
        rumination = 0.84 - 0.011*t + np.random.normal(0, 0.03, n)
        locomotion = 0.88 - 0.010*t + np.random.normal(0, 0.025, n)
        heat       = 0.22 + 0.008*t + np.random.normal(0, 0.025, n)

    else:
        raise ValueError("Unknown regime")

    return pd.DataFrame({
        "activity": np.clip(activity, 0, 1),
        "rumination": np.clip(rumination, 0, 1),
        "locomotion": np.clip(locomotion, 0, 1),
        "heat": np.clip(heat, 0, 1),
    })

def make_trajectory(kind: str):
    if kind == "stable_only":
        parts = [segment("stable", 72)]
        collapse_start = None

    elif kind == "transition_only":
        parts = [segment("stable", 24), segment("transition", 48)]
        collapse_start = None

    elif kind == "to_collapse":
        parts = [segment("stable", 24), segment("transition", 18), segment("collapse", 30)]
        collapse_start = 24 + 18

    else:
        raise ValueError("Unknown trajectory kind")

    df = pd.concat(parts, ignore_index=True)
    return df, collapse_start

# =========================================================
# Structural validity + classifier
# =========================================================

def structural_validity(df: pd.DataFrame) -> float:
    a = df["activity"].values
    r = df["rumination"].values
    l = df["locomotion"].values
    h = df["heat"].values

    slope_a = np.polyfit(np.arange(len(a)), a, 1)[0]
    slope_r = np.polyfit(np.arange(len(r)), r, 1)[0]
    slope_l = np.polyfit(np.arange(len(l)), l, 1)[0]
    slope_h = np.polyfit(np.arange(len(h)), h, 1)[0]

    direction_score = 0
    direction_score += 1 if slope_a < -0.002 else 0
    direction_score += 1 if slope_r < -0.002 else 0
    direction_score += 1 if slope_l < -0.002 else 0
    direction_score += 1 if slope_h >  0.002 else 0
    direction_score /= 4.0

    mid = len(df)//2
    persistence_score = 0
    persistence_score += 1 if a[mid:].mean() < a[:mid].mean() else 0
    persistence_score += 1 if r[mid:].mean() < r[:mid].mean() else 0
    persistence_score += 1 if l[mid:].mean() < l[:mid].mean() else 0
    persistence_score += 1 if h[mid:].mean() > h[:mid].mean() else 0
    persistence_score /= 4.0

    coherence_score = np.mean([
        slope_a < 0,
        slope_r < 0,
        slope_l < 0,
        slope_h > 0
    ])

    return float(np.clip(0.4*direction_score + 0.35*persistence_score + 0.25*coherence_score, 0, 1))

def classify_window(df: pd.DataFrame):
    v = structural_validity(df)

    a_drop = df["activity"].iloc[0] - df["activity"].iloc[-1]
    r_drop = df["rumination"].iloc[0] - df["rumination"].iloc[-1]
    l_drop = df["locomotion"].iloc[0] - df["locomotion"].iloc[-1]
    h_rise = df["heat"].iloc[-1] - df["heat"].iloc[0]

    severity = np.mean([a_drop, r_drop, l_drop, h_rise])

    min_transition_severity = 0.05

    if v < 0.45 or severity < min_transition_severity:
        regime = "stable"
        alert = "GREEN"
    elif severity < 0.22:
        regime = "transition"
        alert = "YELLOW"
    else:
        regime = "collapse"
        alert = "RED"

    return regime, alert, float(v), float(severity)

# =========================================================
# Lead-time validation
# =========================================================

WINDOW = 24
STEP = 3

rows = []

for kind in ["stable_only", "transition_only", "to_collapse"]:
    for run_id in range(60):
        df, collapse_start = make_trajectory(kind)

        first_yellow = None
        first_red = None

        for start in range(0, len(df) - WINDOW + 1, STEP):
            w = df.iloc[start:start+WINDOW]
            regime, alert, v, s = classify_window(w)
            current_t = start + WINDOW

            if first_yellow is None and alert in ("YELLOW", "RED"):
                first_yellow = current_t

            if first_red is None and alert == "RED":
                first_red = current_t

        if collapse_start is not None:
            lead_yellow = None if first_yellow is None else collapse_start - first_yellow
            lead_red = None if first_red is None else collapse_start - first_red
        else:
            lead_yellow = None
            lead_red = None

        rows.append({
            "trajectory_type": kind,
            "first_yellow_time": first_yellow,
            "first_red_time": first_red,
            "collapse_start": collapse_start,
            "lead_time_yellow": lead_yellow,
            "lead_time_red": lead_red,
            "false_yellow_on_stable": int(kind == "stable_only" and first_yellow is not None),
            "false_red_on_stable": int(kind == "stable_only" and first_red is not None),
        })

res = pd.DataFrame(rows)

print("\n=== VALIDATION 4 :: LEAD TIME / EARLY WARNING ===")

stable = res[res["trajectory_type"] == "stable_only"]
to_collapse = res[res["trajectory_type"] == "to_collapse"]

false_yellow_rate = stable["false_yellow_on_stable"].mean()
false_red_rate = stable["false_red_on_stable"].mean()

mean_lead_yellow = to_collapse["lead_time_yellow"].dropna().mean()
mean_lead_red = to_collapse["lead_time_red"].dropna().mean()

det_yellow = to_collapse["lead_time_yellow"].notna().mean()
det_red = to_collapse["lead_time_red"].notna().mean()

print(f"false yellow rate on stable: {false_yellow_rate:.3f}")
print(f"false red rate on stable   : {false_red_rate:.3f}")
print(f"detection rate yellow      : {det_yellow:.3f}")
print(f"detection rate red         : {det_red:.3f}")
print(f"mean lead time yellow      : {mean_lead_yellow:.2f}")
print(f"mean lead time red         : {mean_lead_red:.2f}")

print("\nlead time distributions (to_collapse):")
print(
    to_collapse[["lead_time_yellow", "lead_time_red"]]
    .describe()
    .round(2)
)

res.to_csv("validation4_leadtime_results.csv", index=False)
print("\nSaved: validation4_leadtime_results.csv")

