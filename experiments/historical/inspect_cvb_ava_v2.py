import re
from pathlib import Path
import pandas as pd

root = Path(".")
base = root / "11_real_data" / "cvb_data" / "data" / "cvb_in_ava_format"
exp_dir = root / "11_real_data" / "cvb_data" / "exports"
exp_dir.mkdir(parents=True, exist_ok=True)

train_file = base / "ava_train_set.csv"
val_file   = base / "ava_val_set.csv"
task_train = base / "train_set_tasklist.csv"
task_val   = base / "val_set_tasklist.csv"

train = pd.read_csv(train_file, header=None)
val   = pd.read_csv(val_file, header=None)
task_train_df = pd.read_csv(task_train, header=None)
task_val_df   = pd.read_csv(task_val, header=None)

def extract_ids(series):
    beh = []
    ani = []
    for x in series.astype(str):
        m1 = re.search(r'beh(\d+)', x)
        m2 = re.search(r'ani(\d+)', x)
        beh.append(m1.group(1) if m1 else None)
        ani.append(m2.group(1) if m2 else None)
    return beh, ani

train_beh, train_ani = extract_ids(train[0])
val_beh, val_ani = extract_ids(val[0])

train["behavior_id_from_name"] = train_beh
train["animal_id_from_name"] = train_ani
val["behavior_id_from_name"] = val_beh
val["animal_id_from_name"] = val_ani

beh_counts = pd.concat([train["behavior_id_from_name"], val["behavior_id_from_name"]]).value_counts(dropna=False)
ani_counts = pd.concat([train["animal_id_from_name"], val["animal_id_from_name"]]).value_counts(dropna=False)

train_out = exp_dir / "cvb_ava_train_preview_v2.csv"
val_out   = exp_dir / "cvb_ava_val_preview_v2.csv"
beh_out   = exp_dir / "cvb_behavior_counts_v2.csv"
ani_out   = exp_dir / "cvb_animal_counts_v2.csv"
summary_out = exp_dir / "cvb_ava_summary_v2.txt"

train.head(50).to_csv(train_out, index=False)
val.head(50).to_csv(val_out, index=False)
beh_counts.rename_axis("behavior_id").reset_index(name="count").to_csv(beh_out, index=False)
ani_counts.rename_axis("animal_id").reset_index(name="count").to_csv(ani_out, index=False)

lines = []
lines.append("CVB AVA SUMMARY V2")
lines.append("==================")
lines.append(f"train_rows: {len(train)}")
lines.append(f"val_rows: {len(val)}")
lines.append(f"task_train_shape: {task_train_df.shape}")
lines.append(f"task_val_shape: {task_val_df.shape}")
lines.append("")
lines.append("top_behavior_ids:")
for k, v in beh_counts.head(20).items():
    lines.append(f"- beh{k}: {v}")
lines.append("")
lines.append("top_animal_ids:")
for k, v in ani_counts.head(20).items():
    lines.append(f"- ani{k}: {v}")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB AVA INSPECTION V2 ===")
print(f"train rows         : {len(train)}")
print(f"val rows           : {len(val)}")
print(f"task_train shape   : {task_train_df.shape}")
print(f"task_val shape     : {task_val_df.shape}")

print("\nTop behavior IDs:")
print(beh_counts.head(20).to_string())

print("\nTop animal IDs:")
print(ani_counts.head(20).to_string())

print(f"\nSaved:")
print(f"- {train_out}")
print(f"- {val_out}")
print(f"- {beh_out}")
print(f"- {ani_out}")
print(f"- {summary_out}")

