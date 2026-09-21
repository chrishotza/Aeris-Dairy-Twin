import json
from pathlib import Path
import pandas as pd
from collections import Counter

root = Path(".")
ann_dir = root / "11_real_data" / "cvb_data" / "data" / "annotations"
exp_dir = root / "11_real_data" / "cvb_data" / "exports"
exp_dir.mkdir(parents=True, exist_ok=True)

json_files = list(ann_dir.rglob("*.json"))

rows = []
behavior_counter = Counter()
animal_counter = Counter()

for jf in json_files:
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue

    images = {img.get("id"): img for img in data.get("images", [])}
    anns = data.get("annotations", [])

    for a in anns:
        attrs = a.get("attributes", {}) if isinstance(a.get("attributes", {}), dict) else {}
        behavior = attrs.get("behavior")
        animal_id = attrs.get("id")
        track_id = attrs.get("track_id")
        occluded = attrs.get("occluded")
        keyframe = attrs.get("keyframe")

        image_id = a.get("image_id")
        image_info = images.get(image_id, {})
        file_name = image_info.get("file_name")

        rows.append({
            "source_json": jf.name,
            "image_id": image_id,
            "file_name": file_name,
            "annotation_id": a.get("id"),
            "category_id": a.get("category_id"),
            "behavior": behavior,
            "animal_id": animal_id,
            "track_id": track_id,
            "occluded": occluded,
            "keyframe": keyframe,
            "bbox_x": a.get("bbox", [None, None, None, None])[0],
            "bbox_y": a.get("bbox", [None, None, None, None])[1],
            "bbox_w": a.get("bbox", [None, None, None, None])[2],
            "bbox_h": a.get("bbox", [None, None, None, None])[3],
            "area": a.get("area")
        })

        if behavior is not None:
            behavior_counter[str(behavior)] += 1
        if animal_id is not None:
            animal_counter[str(animal_id)] += 1

beh_df = pd.DataFrame(rows)
full_out = exp_dir / "cvb_behavior_table_v1.csv"
beh_df.to_csv(full_out, index=False)

count_out = exp_dir / "cvb_behavior_counts_full_v1.csv"
pd.DataFrame(
    [{"behavior": k, "count": v} for k, v in behavior_counter.most_common()]
).to_csv(count_out, index=False)

animal_out = exp_dir / "cvb_animal_id_counts_full_v1.csv"
pd.DataFrame(
    [{"animal_id": k, "count": v} for k, v in animal_counter.most_common()]
).to_csv(animal_out, index=False)

summary_out = exp_dir / "cvb_behavior_summary_v1.txt"
lines = []
lines.append("CVB BEHAVIOR SUMMARY V1")
lines.append("=======================")
lines.append(f"json_files: {len(json_files)}")
lines.append(f"annotation_rows: {len(beh_df)}")
lines.append(f"unique_behaviors: {beh_df['behavior'].dropna().nunique()}")
lines.append(f"unique_animal_ids: {beh_df['animal_id'].dropna().nunique()}")
lines.append("")
lines.append("top_behaviors:")
for k, v in behavior_counter.most_common(20):
    lines.append(f"- {k}: {v}")
summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB BEHAVIOR EXTRACTION V1 ===")
print(f"json files       : {len(json_files)}")
print(f"annotation rows  : {len(beh_df)}")
print(f"unique behaviors : {beh_df['behavior'].dropna().nunique()}")
print(f"unique animal_ids: {beh_df['animal_id'].dropna().nunique()}")

print("\nTop behaviors:")
print(pd.DataFrame([{"behavior": k, "count": v} for k, v in behavior_counter.most_common(20)]).to_string(index=False))

print(f"\nSaved:")
print(f"- {full_out}")
print(f"- {count_out}")
print(f"- {animal_out}")
print(f"- {summary_out}")

