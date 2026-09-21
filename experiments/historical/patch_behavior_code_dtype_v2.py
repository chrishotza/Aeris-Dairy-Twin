from pathlib import Path

path = Path(r".\10_code\run_cvb_param_sweep_v2.py")
text = path.read_text(encoding="utf-8")

old = 'calib["behavior_code"] = calib["behavior_code"].astype(str)\nmain = backfill_by_key(main, calib, "behavior_code", ["calibration_class"])'
new = 'calib["behavior_code"] = calib["behavior_code"].astype(str)\nmain["behavior_code"] = main["behavior_code"].astype(str)\nmain = backfill_by_key(main, calib, "behavior_code", ["calibration_class"])'

if old not in text:
    raise SystemExit("No encontré el bloque exacto para parchar. No hice cambios.")

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("Patched:", path)

