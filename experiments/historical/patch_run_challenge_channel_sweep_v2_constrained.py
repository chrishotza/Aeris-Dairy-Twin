from pathlib import Path

path = Path(r".\10_code\run_challenge_channel_sweep_v1.py")
text = path.read_text(encoding="utf-8")

# 1) bajar el peso del recall bruto y poner restricciones duras
old_block = """    score = (
        2.4 * mean_recall +
        2.4 * min_recall +
        0.9 * mean_precision +
        0.8 * mean_specificity +
        0.6 * overall["accuracy"] +
        0.8 * float(lame["recall"]) +
        0.8 * float(vent["recall"]) +
        0.5 * float(heat["recall"])
    )

    score -= 2.0 * float(baseline["concern_rate"])
    score -= 1.0 * (1.0 - float(baseline["specificity"]))

    if overall["concern_rate"] > 0.58:
        score -= 2.8 * (overall["concern_rate"] - 0.58)
    if overall["red_rate"] > 0.30:
        score -= 3.2 * (overall["red_rate"] - 0.30)
"""

new_block = """    # hard viability gates first
    baseline_concern = float(baseline["concern_rate"])
    baseline_spec = float(baseline["specificity"])
    overall_concern = float(overall["concern_rate"])
    overall_red = float(overall["red_rate"])
    overall_spec = float(overall["specificity"])

    viable = True
    if baseline_concern > 0.08:
        viable = False
    if baseline_spec < 0.92:
        viable = False
    if overall_concern > 0.50:
        viable = False
    if overall_red > 0.12:
        viable = False
    if overall_spec < 0.72:
        viable = False

    score = (
        1.8 * mean_recall +
        2.0 * min_recall +
        1.1 * mean_precision +
        1.0 * mean_specificity +
        0.7 * overall["accuracy"] +
        1.0 * float(lame["recall"]) +
        1.0 * float(vent["recall"]) +
        0.6 * float(heat["recall"])
    )

    score -= 1.6 * baseline_concern
    score -= 0.8 * (1.0 - baseline_spec)

    if overall_concern > 0.42:
        score -= 2.0 * (overall_concern - 0.42)
    if overall_red > 0.08:
        score -= 2.5 * (overall_red - 0.08)

    if not viable:
        score -= 100.0
"""

if old_block not in text:
    raise SystemExit("No encontré el bloque objetivo para reemplazar.")

text = text.replace(old_block, new_block, 1)

# 2) bajar un poco el espacio de búsqueda para iterar más rápido
text = text.replace("N_RANDOM = 1200", "N_RANDOM = 1000", 1)
text = text.replace("TOP_SEEDS = 100", "TOP_SEEDS = 80", 1)
text = text.replace("N_REFINE = 2600", "N_REFINE = 2200", 1)

# 3) cambiar nombre de outputs para no pisar la versión mala
text = text.replace("challenge_channel_sweep_v1", "challenge_channel_sweep_v2_constrained")
text = text.replace("challenge_channel_group_predictions_v1.csv", "challenge_channel_group_predictions_v2.csv")
text = text.replace("challenge_channel_unit_predictions_v1.csv", "challenge_channel_unit_predictions_v2.csv")
text = text.replace("challenge_channel_scenarios_v1.csv", "challenge_channel_scenarios_v2.csv")
text = text.replace("challenge_channel_dashboard_summary_v1.csv", "challenge_channel_dashboard_summary_v2.csv")
text = text.replace("challenge_channel_best_params_v1.json", "challenge_channel_best_params_v2.json")
text = text.replace("challenge_channel_sweep_summary_v1.txt", "challenge_channel_sweep_summary_v2.txt")
text = text.replace("=== CHALLENGE CHANNEL SWEEP V1 ===", "=== CHALLENGE CHANNEL SWEEP V2 CONSTRAINED ===")
text = text.replace("CHALLENGE CHANNEL SWEEP V1", "CHALLENGE CHANNEL SWEEP V2 CONSTRAINED")

new_path = Path(r".\10_code\run_challenge_channel_sweep_v2_constrained.py")
new_path.write_text(text, encoding="utf-8")
print(f"Wrote: {new_path}")

