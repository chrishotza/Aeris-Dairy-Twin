import numpy as np
import pandas as pd

np.random.seed(42)

# =========================================================
# Synthetic generator
# =========================================================

def generate_animal_series(regime: str, n: int = 48):
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

    df = pd.DataFrame({
        "activity": np.clip(activity, 0, 1),
        "rumination": np.clip(rumination, 0, 1),
        "locomotion": np.clip(locomotion, 0, 1),
        "heat": np.clip(heat, 0, 1),
    })
    return df

# =========================================================
# Structural validity
# =========================================================

def structural_validity(df: pd.DataFrame) -> float:
    # deterioration signals
    a = df["activity"].values
    r = df["rumination"].values
    l = df["locomotion"].values
    h = df["heat"].values

    # slopes
    slope_a = np.polyfit(np.arange(len(a)), a, 1)[0]
    slope_r = np.polyfit(np.arange(len(r)), r, 1)[0]
    slope_l = np.polyfit(np.arange(len(l)), l, 1)[0]
    slope_h = np.polyfit(np.arange(len(h)), h, 1)[0]

    # deterioration direction:
    # activity, rumination, locomotion should go DOWN
    # heat should go UP
    direction_score = 0
    direction_score += 1 if slope_a < -0.002 else 0
    direction_score += 1 if slope_r < -0.002 else 0
    direction_score += 1 if slope_l < -0.002 else 0
    direction_score += 1 if slope_h >  0.002 else 0
    direction_score /= 4.0

    # persistence: compare first half vs second half
    mid = len(df)//2
    persistence_score = 0
    persistence_score += 1 if a[mid:].mean() < a[:mid].mean() else 0
    persistence_score += 1 if r[mid:].mean() < r[:mid].mean() else 0
    persistence_score += 1 if l[mid:].mean() < l[:mid].mean() else 0
    persistence_score += 1 if h[mid:].mean() > h[:mid].mean() else 0
    persistence_score /= 4.0

    # cross-signal coherence: slopes aligned with expected deterioration
    signs = np.array([
        slope_a < 0,
        slope_r < 0,
        slope_l < 0,
        slope_h > 0
    ], dtype=float)
    coherence_score = signs.mean()

    validity = 0.4*direction_score + 0.35*persistence_score + 0.25*coherence_score
    return float(np.clip(validity, 0, 1))

# =========================================================
# Regime classifier
# =========================================================

def classify_regime(df: pd.DataFrame):
    v = structural_validity(df)

    a_drop = df["activity"].iloc[0] - df["activity"].iloc[-1]
    r_drop = df["rumination"].iloc[0] - df["rumination"].iloc[-1]
    l_drop = df["locomotion"].iloc[0] - df["locomotion"].iloc[-1]
    h_rise = df["heat"].iloc[-1] - df["heat"].iloc[0]

    severity = np.mean([a_drop, r_drop, l_drop, h_rise])

    if v < 0.45:
        pred = "stable"
    elif severity < 0.22:
        pred = "transition"
    else:
        pred = "collapse"

    return pred, v, severity

# =========================================================
# Validation run
# =========================================================

rows = []

for regime in ["stable", "transition", "collapse"]:
    for i in range(50):
        df = generate_animal_series(regime, n=48)
        pred, validity, severity = classify_regime(df)
        rows.append({
            "true_regime": regime,
            "pred_regime": pred,
            "validity_score": round(validity, 4),
            "severity_score": round(float(severity), 4),
            "correct": int(regime == pred)
        })

res = pd.DataFrame(rows)

acc = res["correct"].mean()
cm = pd.crosstab(res["true_regime"], res["pred_regime"])

print("\n=== VALIDATION 1 :: STRUCTURAL VALIDITY + REGIME ===")
print(f"accuracy: {acc:.3f}")
print("\nconfusion matrix:")
print(cm)
print("\nmean validity by true regime:")
print(res.groupby("true_regime")["validity_score"].mean().round(3))
print("\nmean severity by true regime:")
print(res.groupby("true_regime")["severity_score"].mean().round(3))

res.to_csv("validation1_results.csv", index=False)
print("\nSaved: validation1_results.csv")
