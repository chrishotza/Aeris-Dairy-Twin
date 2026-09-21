import json
from pathlib import Path
import pandas as pd
from collections import Counter

root = Path(".")
base = root / "11_real_data" / "cvb_data"
data_dir = base / "data"
exp_dir = base / "exports"
exp_dir.mkdir(parents=True, exist_ok=True)

json_files = list(data_dir.rglob("*.json"))
csv_files  = list(data_dir.rglob("*.csv"))
pkl_files  = list(data_dir.rglob("*.pkl"))

tree_rows = []
for p in data_dir.rglob("*"):
    if p.is_file():
        tree_rows.append({
            "relative_path": str(p.relative_to(base)),
            "extension": p.suffix.lower(),
            "size_bytes": p.stat().st_size
        })

tree_df = pd.DataFrame(tree_rows).sort_values(["extension", "relative_path"]).reset_index(drop=True)
tree_out = exp_dir / "cvb_data_tree_v2.csv"
tree_df.to_csv(tree_out, index=False)

json_rows = []
top_key_counter = Counter()
string_counter = Counter()

for jf in json_files[:80]:  # sample first 80 JSONs for structure scan
    try:
        text = jf.read_text(encoding="utf-8")
        data = json.loads(text)
        status = "ok"
    except Exception:
        json_rows.append({
            "relative_path": str(jf.relative_to(base)),
            "status": "unreadable",
            "top_keys": "",
            "n_top_keys": None
        })
        continue

    keys = list(data.keys()) if isinstance(data, dict) else []
    for k in keys:
        top_key_counter[k] += 1

    # shallow string scan
    found_strings = set()
    def walk(x, depth=0):
        if depth > 3:
            return
        if isinstance(x, dict):
            for _, v in x.items():
                walk(v, depth+1)
        elif isinstance(x, list):
            for v in x[:50]:
                walk(v, depth+1)
        elif isinstance(x, str):
            s = x.strip()
            if 0 < len(s) <= 40:
                found_strings.add(s)

    walk(data)

    for s in found_strings:
        string_counter[s] += 1

    json_rows.append({
        "relative_path": str(jf.relative_to(base)),
        "status": status,
        "top_keys": ";".join(keys[:20]),
        "n_top_keys": len(keys)
    })

json_df = pd.DataFrame(json_rows)
json_out = exp_dir / "cvb_json_manifest_v2.csv"
json_df.to_csv(json_out, index=False)

top_keys_out = exp_dir / "cvb_top_keys_v2.csv"
pd.DataFrame(
    [{"key": k, "count": v} for k, v in top_key_counter.most_common(50)]
).to_csv(top_keys_out, index=False)

strings_out = exp_dir / "cvb_detected_strings_v2.csv"
pd.DataFrame(
    [{"value": k, "count": v} for k, v in string_counter.most_common(100)]
).to_csv(strings_out, index=False)

summary_lines = []
summary_lines.append("CVB INSPECTION V2")
summary_lines.append("=================")
summary_lines.append(f"json_files: {len(json_files)}")
summary_lines.append(f"csv_files: {len(csv_files)}")
summary_lines.append(f"pkl_files: {len(pkl_files)}")
summary_lines.append(f"tree_csv: {tree_out}")
summary_lines.append(f"json_manifest_csv: {json_out}")
summary_lines.append(f"top_keys_csv: {top_keys_out}")
summary_lines.append(f"strings_csv: {strings_out}")

summary_out = exp_dir / "cvb_inspection_summary_v2.txt"
summary_out.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n=== CVB INSPECTION V2 ===")
print(f"json files : {len(json_files)}")
print(f"csv files  : {len(csv_files)}")
print(f"pkl files  : {len(pkl_files)}")

print("\nTop 10 JSON top-level keys:")
for k, v in top_key_counter.most_common(10):
    print(f"- {k}: {v}")

print("\nTop 20 detected strings:")
for k, v in string_counter.most_common(20):
    print(f"- {k}: {v}")

print(f"\nSaved:")
print(f"- {tree_out}")
print(f"- {json_out}")
print(f"- {top_keys_out}")
print(f"- {strings_out}")
print(f"- {summary_out}")

