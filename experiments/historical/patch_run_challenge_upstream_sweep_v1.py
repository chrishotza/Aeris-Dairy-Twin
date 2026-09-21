from pathlib import Path

path = Path(r".\10_code\run_challenge_upstream_sweep_v1.py")
text = path.read_text(encoding="utf-8")

replacements = {
'''group_keys = pd.factorize(
    list(zip(df["run_id"], df["scenario"], df["hour"], df["unit_id"], df["group_id"]))
)[0]''':
'''group_keys = pd.factorize(
    pd.MultiIndex.from_frame(df[["run_id", "scenario", "hour", "unit_id", "group_id"]])
)[0]''',

'''unit_keys = pd.factorize(
    list(zip(df["run_id"], df["scenario"], df["hour"], df["unit_id"]))
)[0]''':
'''unit_keys = pd.factorize(
    pd.MultiIndex.from_frame(df[["run_id", "scenario", "hour", "unit_id"]])
)[0]''',

'''unit_order_idx = pd.factorize(list(zip(unit_meta["run_id"], unit_meta["scenario"], unit_meta["unit_id"])))[0]''':
'''unit_order_idx = pd.factorize(
    pd.MultiIndex.from_frame(unit_meta[["run_id", "scenario", "unit_id"]])
)[0]''',

'''unit_idx_from_meta = pd.factorize(list(zip(unit_meta["run_id"], unit_meta["scenario"], unit_meta["hour"], unit_meta["unit_id"])))[0]''':
'''unit_idx_from_meta = pd.factorize(
    pd.MultiIndex.from_frame(unit_meta[["run_id", "scenario", "hour", "unit_id"]])
)[0]''',

'''N_RANDOM = 2500''': '''N_RANDOM = 1200''',
'''TOP_SEEDS = 120''': '''TOP_SEEDS = 80''',
'''N_REFINE = 5000''': '''N_REFINE = 2400''',
'''PROGRESS_EVERY = 250''': '''PROGRESS_EVERY = 100''',
}

changed = 0
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
        changed += 1

path.write_text(text, encoding="utf-8")
print(f"Patched {path} | replacements applied: {changed}")

