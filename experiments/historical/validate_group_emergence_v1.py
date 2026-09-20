import numpy as np
import pandas as pd

np.random.seed(123)

# =========================================================
# Animal generator
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

    return pd.DataFrame({
        "activity": np.clip(activity, 0, 1),
        "rumination": np.clip(rumination, 0, 1),
        "locomotion": np.clip(locomotion, 0, 1),
        "heat": np.clip(heat, 0, 1),
    })

# =========================================================
# Structural validity + regime
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

def classify_animal(df: pd.DataFrame):
    v = structural_validity(df)

    a_drop = df["activity"].iloc[0] - df["activity"].iloc[-1]
    r_drop = df["rumination"].iloc[0] - df["rumination"].iloc[-1]
    l_drop = df["locomotion"].iloc[0] - df["locomotion"].iloc[-1]
    h_rise = df["heat"].iloc[-1] - df["heat"].iloc[0]

    severity = np.mean([a_drop, r_drop, l_drop, h_rise])

    if v < 0.45:
        regime = "stable"
    elif severity < 0.22:
        regime = "transition"
    else:
        regime = "collapse"

    return {
        "validity_score": float(v),
        "severity_score": float(severity),
        "animal_regime": regime
    }

# =========================================================
# Group simulator
# =========================================================

def simulate_group(true_group_regime: str, n_animals: int = 20):
    animal_rows = []

    if true_group_regime == "stable":
        mix = ["stable"] * n_animals

    elif true_group_regime == "transition":
        # grupo en transición: mezcla con varios animales deteriorando
        mix = ["stable"] * 10 + ["transition"] * 8 + ["collapse"] * 2
        np.random.shuffle(mix)

    elif true_group_regime == "collapse":
        # grupo en colapso: muchos animales mal
        mix = ["stable"] * 4 + ["transition"] * 6 + ["collapse"] * 10
        np.random.shuffle(mix)

    else:
        raise ValueError("Unknown true_group_regime")

    for i, reg in enumerate(mix):
        df = generate_animal_series(reg, n=48)
        out = classify_animal(df)
        animal_rows.append({
            "animal_id": f"A{i+1:03d}",
            "true_animal_regime": reg,
            **out
        })

    animals = pd.DataFrame(animal_rows)

    # -------- group logic --------
    n_stable = (animals["animal_regime"] == "stable").sum()
    n_trans  = (animals["animal_regime"] == "transition").sum()
    n_coll   = (animals["animal_regime"] == "collapse").sum()

    mean_validity = animals["validity_score"].mean()
    mean_severity = animals["severity_score"].mean()

    # group escalation rule
    if n_coll >= 6 or (n_coll >= 4 and mean_severity > 0.24):
        group_regime = "collapse"
    elif (n_trans + n_coll) >= 6 and mean_validity > 0.65:
        group_regime = "transition"
    else:
        group_regime = "stable"

    return {
        "true_group_regime": true_group_regime,
        "pred_group_regime": group_regime,
        "animals_stable": int(n_stable),
        "animals_transition": int(n_trans),
        "animals_collapse": int(n_coll),
        "mean_validity": round(float(mean_validity), 4),
        "mean_severity": round(float(mean_severity), 4),
        "correct": int(true_group_regime == group_regime)
    }

# =========================================================
# Validation 2
# =========================================================

rows = []
for regime in ["stable", "transition", "collapse"]:
    for _ in range(40):
        rows.append(simulate_group(regime, n_animals=20))

res = pd.DataFrame(rows)

acc = res["correct"].mean()
cm = pd.crosstab(res["true_group_regime"], res["pred_group_regime"])

print("\n=== VALIDATION 2 :: GROUP EMERGENCE ===")
print(f"accuracy: {acc:.3f}")
print("\nconfusion matrix:")
print(cm)

print("\nmean counts by true group regime:")
print(
    res.groupby("true_group_regime")[["animals_stable", "animals_transition", "animals_collapse", "mean_validity", "mean_severity"]]
    .mean()
    .round(3)
)

res.to_csv("validation2_group_results.csv", index=False)
print("\nSaved: validation2_group_results.csv")

