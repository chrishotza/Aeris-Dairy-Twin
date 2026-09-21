import json
from pathlib import Path
import pandas as pd
from collections import Counter

root = Path(".")
base = root / "11_real_data" / "cvb_data" / "data"
exp_dir = root / "11_real_data" / "cvb_data" / "exports"
exp_dir.mkdir(parents=True, exist_ok=True)

json_files = list(base.rglob("*.json"))

category_rows = []
ann_counter = Counter()
image_counter = Counter()
file_rows = []

for jf in json_files:
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue

    if not isinstance(data, dict):
        continue

    cats = data.get("categories", [])
    imgs = data.get("images", [])
    anns = data.get("annotations", [])

    # map category_id -> category_name
    cat_map = {}
    for c in cats:
        cid = c.get("id")
        cname = c.get("name", f"cat_{cid}")
        cat_map[cid] = cname
        category_rows.append({
            "source_json": str(jf.relative_to(base)),
            "category_id": cid,
            "category_name": cname,
            "supercategory": c.get("supercategory", "")
        })

    for a in anns:
        cid = a.get("category_id")
        cname = cat_map.get(cid, f"cat_{cid}")
        ann_counter[cname] += 1

    for _ in imgs:
        for cname in cat_map.values():
            image_counter[cname] += 0

    file_rows.append({
        "source_json": str(jf.relative_to(base)),
        "n_categories": len(cats),
        "n_images": len(imgs),
        "n_annotations": len(anns)
    })

cats_df = pd.DataFrame(category_rows).drop_duplicates().sort_values(
    ["category_name", "source_json"]
).reset_index(drop=True)

file_df = pd.DataFrame(file_rows).sort_values(
    ["n_annotations", "n_images"], ascending=False
).reset_index(drop=True)

ann_df = pd.DataFrame(
    [{"category_name": k, "annotation_count": v} for k, v in ann_counter.items()]
).sort_values("annotation_count", ascending=False).reset_index(drop=True)

cats_out = exp_dir / "cvb_categories_v1.csv"
ann_out = exp_dir / "cvb_annotation_counts_v1.csv"
file_out = exp_dir / "cvb_annotation_files_v1.csv"
summary_out = exp_dir / "cvb_categories_summary_v1.txt"

cats_df.to_csv(cats_out, index=False)
ann_df.to_csv(ann_out, index=False)
file_df.to_csv(file_out, index=False)

lines = []
lines.append("CVB CATEGORIES SUMMARY V1")
lines.append("=========================")
lines.append(f"json_files_scanned: {len(json_files)}")
lines.append(f"unique_categories: {cats_df['category_name'].nunique() if len(cats_df) else 0}")
lines.append("")
lines.append("top_categories_by_annotation_count:")
if len(ann_df) == 0:
    lines.append("- none")
else:
    for _, row in ann_df.head(20).iterrows():
        lines.append(f"- {row['category_name']}: {row['annotation_count']}")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB CATEGORY EXTRACTION V1 ===")
print(f"json files scanned : {len(json_files)}")
print(f"unique categories  : {cats_df['category_name'].nunique() if len(cats_df) else 0}")

print("\nTop categories by annotation count:")
if len(ann_df) == 0:
    print("none")
else:
    print(ann_df.head(20).to_string(index=False))

print(f"\nSaved:")
print(f"- {cats_out}")
print(f"- {ann_out}")
print(f"- {file_out}")
print(f"- {summary_out}")

