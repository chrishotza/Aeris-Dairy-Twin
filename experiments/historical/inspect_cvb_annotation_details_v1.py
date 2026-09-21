import json
from pathlib import Path
import pandas as pd
from collections import Counter

root = Path(".")
ann_dir = root / "11_real_data" / "cvb_data" / "data" / "annotations"
exp_dir = root / "11_real_data" / "cvb_data" / "exports"
exp_dir.mkdir(parents=True, exist_ok=True)

json_files = list(ann_dir.rglob("*.json"))

ann_key_counter = Counter()
attr_key_counter = Counter()
attr_val_counter = Counter()
sample_lines = []

scanned = 0
for jf in json_files[:40]:
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue

    anns = data.get("annotations", [])
    if not anns:
        continue

    scanned += 1
    sample_lines.append(f"=== FILE: {jf.name} ===")
    sample_lines.append(f"n_annotations: {len(anns)}")

    for a in anns[:10]:
        for k in a.keys():
            ann_key_counter[k] += 1

        # buscar campos potenciales de comportamiento
        for k, v in a.items():
            if isinstance(v, dict):
                attr_key_counter[k] += 1
                for kk, vv in v.items():
                    attr_key_counter[f"{k}.{kk}"] += 1
                    if isinstance(vv, (str, int, float)):
                        attr_val_counter[f"{k}.{kk}={vv}"] += 1
            elif isinstance(v, list):
                attr_key_counter[k] += 1
                for vv in v[:10]:
                    if isinstance(vv, (str, int, float)):
                        attr_val_counter[f"{k}={vv}"] += 1
            elif isinstance(v, (str, int, float)):
                attr_val_counter[f"{k}={v}"] += 1

        sample_lines.append(str(a))

    sample_lines.append("")

ann_keys_out = exp_dir / "cvb_annotation_keys_v1.csv"
attr_keys_out = exp_dir / "cvb_annotation_attr_keys_v1.csv"
attr_vals_out = exp_dir / "cvb_annotation_attr_values_v1.csv"
sample_out = exp_dir / "cvb_annotation_samples_v1.txt"

pd.DataFrame(
    [{"key": k, "count": v} for k, v in ann_key_counter.most_common(100)]
).to_csv(ann_keys_out, index=False)

pd.DataFrame(
    [{"key": k, "count": v} for k, v in attr_key_counter.most_common(100)]
).to_csv(attr_keys_out, index=False)

pd.DataFrame(
    [{"value": k, "count": v} for k, v in attr_val_counter.most_common(200)]
).to_csv(attr_vals_out, index=False)

sample_out.write_text("\n".join(sample_lines), encoding="utf-8")

print("\n=== CVB ANNOTATION DEEP INSPECTION V1 ===")
print(f"json files found : {len(json_files)}")
print(f"json files scanned with annotations : {scanned}")

print("\nTop annotation keys:")
for k, v in ann_key_counter.most_common(20):
    print(f"- {k}: {v}")

print("\nTop nested/attribute keys:")
for k, v in attr_key_counter.most_common(20):
    print(f"- {k}: {v}")

print("\nTop attribute-like values:")
for k, v in attr_val_counter.most_common(30):
    print(f"- {k}: {v}")

print(f"\nSaved:")
print(f"- {ann_keys_out}")
print(f"- {attr_keys_out}")
print(f"- {attr_vals_out}")
print(f"- {sample_out}")

