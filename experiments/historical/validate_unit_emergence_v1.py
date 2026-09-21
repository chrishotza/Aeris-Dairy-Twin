import numpy as np
import pandas as pd

np.random.seed(999)

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
# Structural validity + animal regime
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
# Group layer
# =========================================================

def simulate_group(true_group_regime: str, n_animals: int = 20):
    if true_group_regime == "stable":
        mix = ["stable"] * n_animals
    elif true_group_regime == "transition":
        mix = ["stable"] * 10 + ["transition"] * 8 + ["collapse"] * 2
        np.random.shuffle(mix)
    elif true_group_regime == "collapse":
        mix = ["stable"] * 4 + ["transition"] * 6 + ["collapse"] * 10
        np.random.shuffle(mix)
    else:
        raise ValueError("Unknown true_group_regime")

    animal_rows = []
    for i, reg in enumerate(mix):
        df = generate_animal_series(reg, n=48)
        out = classify_animal(df)
        animal_rows.append({
            "animal_id": f"A{i+1:03d}",
            "true_animal_regime": reg,
            **out
        })

    animals = pd.DataFrame(animal_rows)

    n_stable = (animals["animal_regime"] == "stable").sum()
    n_trans  = (animals["animal_regime"] == "transition").sum()
    n_coll   = (animals["animal_regime"] == "collapse").sum()

    mean_validity = animals["validity_score"].mean()
    mean_severity = animals["severity_score"].mean()

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
        "mean_validity": float(mean_validity),
        "mean_severity": float(mean_severity),
    }

# =========================================================
# Unit / farm layer
# =========================================================

def simulate_unit(true_unit_regime: str, n_groups: int = 4):
    if true_unit_regime == "stable":
        group_mix = ["stable"] * n_groups

    elif true_unit_regime == "transition":
        group_mix = ["stable", "stable", "transition", "transition"]

    elif true_unit_regime == "collapse":
        group_mix = ["stable", "transition", "collapse", "collapse"]

    else:
        raise ValueError("Unknown true_unit_regime")

    np.random.shuffle(group_mix)

    groups = []
    for i, g_reg in enumerate(group_mix):
        g = simulate_group(g_reg, n_animals=20)
        g["group_id"] = f"G{i+1:02d}"
        groups.append(g)

    gdf = pd.DataFrame(groups)

    pred_stable = (gdf["pred_group_regime"] == "stable").sum()
    pred_trans  = (gdf["pred_group_regime"] == "transition").sum()
    pred_coll   = (gdf["pred_group_regime"] == "collapse").sum()

    mean_validity = gdf["mean_validity"].mean()
    mean_severity = gdf["mean_severity"].mean()

    if pred_coll >= 2 or (pred_coll >= 1 and pred_trans >= 2):
        pred_unit_regime = "collapse"
        alert_level = "RED"
    elif (pred_trans + pred_coll) >= 2 and mean_validity > 0.60:
        pred_unit_regime = "transition"
        alert_level = "YELLOW"
    else:
        pred_unit_regime = "stable"
        alert_level = "GREEN"

    return {
        "true_unit_regime": true_unit_regime,
        "pred_unit_regime": pred_unit_regime,
        "alert_level": alert_level,
        "groups_stable": int(pred_stable),
        "groups_transition": int(pred_trans),
        "groups_collapse": int(pred_coll),
        "mean_group_validity": round(float(mean_validity), 4),
        "mean_group_severity": round(float(mean_severity), 4),
        "correct": int(true_unit_regime == pred_unit_regime)
    }

# =========================================================
# Validation 3
# =========================================================

rows = []
for regime in ["stable", "transition", "collapse"]:
    for _ in range(50):
        rows.append(simulate_unit(regime, n_groups=4))

res = pd.DataFrame(rows)

acc = res["correct"].mean()
cm = pd.crosstab(res["true_unit_regime"], res["pred_unit_regime"])

print("\n=== VALIDATION 3 :: UNIT / FARM EMERGENCE ===")
print(f"accuracy: {acc:.3f}")
print("\nconfusion matrix:")
print(cm)

print("\nmean counts by true unit regime:")
print(
    res.groupby("true_unit_regime")[["groups_stable", "groups_transition", "groups_collapse", "mean_group_validity", "mean_group_severity"]]
    .mean()
    .round(3)
)

print("\nalert levels by true unit regime:")
print(pd.crosstab(res["true_unit_regime"], res["alert_level"]))

res.to_csv("validation3_unit_results.csv", index=False)
print("\nSaved: validation3_unit_results.csv")

