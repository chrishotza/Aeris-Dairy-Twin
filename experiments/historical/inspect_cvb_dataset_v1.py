import json
from pathlib import Path
import pandas as pd

root = Path(".")
base = root / "11_real_data" / "cvb_data"
raw_dir = base / "raw_frames"
ann_dir = base / "annotations"
exp_dir = base / "exports"

ann_files = list(ann_dir.rglob("*.json"))
clip_dirs = [p for p in raw_dir.iterdir() if p.is_dir()] if raw_dir.exists() else []

rows = []
all_labels = {}

for jf in ann_files:
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        rows.append({
            "file_name": jf.name,
            "status": "unreadable_json",
            "n_top_keys": None,
            "n_detected_labels": None
        })
        continue

    labels_found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (str, int, float)) and isinstance(v, str):
                    labels_found.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)

    for lb in labels_found:
        all_labels[lb] = all_labels.get(lb, 0) + 1

    rows.append({
        "file_name": jf.name,
        "status": "ok",
        "n_top_keys": len(data.keys()) if isinstance(data, dict) else None,
        "n_detected_labels": len(labels_found)
    })

summary_df = pd.DataFrame(rows)
summary_out = exp_dir / "cvb_annotation_manifest_v1.csv"
summary_df.to_csv(summary_out, index=False)

labels_out = exp_dir / "cvb_detected_labels_v1.csv"
pd.DataFrame(
    [{"label": k, "count": v} for k, v in sorted(all_labels.items(), key=lambda x: (-x[1], x[0]))]
).to_csv(labels_out, index=False)

txt = []
txt.append("CVB INSPECTION V1")
txt.append("=================")
txt.append(f"clip_folders_found: {len(clip_dirs)}")
txt.append(f"annotation_json_found: {len(ann_files)}")
txt.append(f"manifest_csv: {summary_out}")
txt.append(f"labels_csv: {labels_out}")

(exp_dir / "cvb_inspection_summary_v1.txt").write_text("\n".join(txt), encoding="utf-8")

print("\n=== CVB INSPECTION V1 ===")
print(f"clip folders found   : {len(clip_dirs)}")
print(f"annotation json found: {len(ann_files)}")
print(f"saved manifest       : {summary_out}")
print(f"saved labels         : {labels_out}")

