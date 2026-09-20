import numpy as np
import pandas as pd

np.random.seed(2027)

# =========================================================
# Synthetic trajectory generator
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
        event_start = None
        event_type = "none"

    elif kind == "transition_only":
        parts = [segment("stable", 24), segment("transition", 48)]
        event_start = 24
        event_type = "transition"

    elif kind == "to_collapse":
        parts = [segment("stable", 24), segment("transition", 18), segment("collapse", 30)]
        event_start = 24
        event_type = "collapse"

    else:
        raise ValueError("Unknown trajectory kind")

    df = pd.concat(parts, ignore_index=True)
    return df, event_start, event_type

# =========================================================
# Core scoring
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
        return "GREEN", v, severity
    elif severity < 0.22:
        return "YELLOW", v, severity
    else:
        return "RED", v, severity

# =========================================================
# Alert logic evaluation
# =========================================================

WINDOW = 24
STEP = 3

rows = []

for kind in ["stable_only", "transition_only", "to_collapse"]:
    for run_id in range(80):
        df, event_start, event_type = make_trajectory(kind)

        alerts = []
        for start in range(0, len(df) - WINDOW + 1, STEP):
            w = df.iloc[start:start+WINDOW]
            alert, v, s = classify_window(w)
            t = start + WINDOW
            alerts.append((t, alert, v, s))

        first_yellow = next((t for t,a,_,_ in alerts if a == "YELLOW"), None)
        first_red    = next((t for t,a,_,_ in alerts if a == "RED"), None)

        has_event = event_type != "none"
        pred_positive = first_yellow is not None

        tp = int(has_event and pred_positive)
        fp = int((not has_event) and pred_positive)
        fn = int(has_event and (not pred_positive))
        tn = int((not has_event) and (not pred_positive))

        rows.append({
            "trajectory_type": kind,
            "event_type": event_type,
            "first_yellow": first_yellow,
            "first_red": first_red,
            "event_start": event_start,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n_alert_windows": sum(1 for _,a,_,_ in alerts if a != "GREEN"),
        })

res = pd.DataFrame(rows)

tp = res["tp"].sum()
fp = res["fp"].sum()
fn = res["fn"].sum()
tn = res["tn"].sum()

precision = tp / (tp + fp + 1e-12)
recall    = tp / (tp + fn + 1e-12)
specificity = tn / (tn + fp + 1e-12)
f1 = 2*precision*recall / (precision + recall + 1e-12)

print("\n=== VALIDATION 5 :: ALERT QUALITY ===")
print(f"precision   : {precision:.3f}")
print(f"recall      : {recall:.3f}")
print(f"specificity : {specificity:.3f}")
print(f"f1 score    : {f1:.3f}")

print("\nmean number of alert windows by trajectory:")
print(res.groupby("trajectory_type")["n_alert_windows"].mean().round(2))

print("\nfirst alert availability:")
print(
    res.groupby("trajectory_type")[["first_yellow", "first_red"]]
    .apply(lambda x: x.notna().mean())
    .round(3)
)

res.to_csv("validation5_alert_quality_results.csv", index=False)
print("\nSaved: validation5_alert_quality_results.csv")
