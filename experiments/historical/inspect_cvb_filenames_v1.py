import pandas as pd
from pathlib import Path

root = Path(".")
inp = root / "11_real_data" / "cvb_data" / "exports" / "cvb_behavior_table_v1.csv"
out = root / "11_real_data" / "cvb_data" / "exports" / "cvb_filename_inspection_v1.txt"

df = pd.read_csv(inp, dtype={"animal_id": str, "track_id": str})
df["file_name"] = df["file_name"].fillna("missing").astype(str)

sample = df["file_name"].drop_duplicates().sort_values().head(200).tolist()

lines = []
lines.append("CVB FILE NAME INSPECTION V1")
lines.append("===========================")
lines.append(f"unique_file_names: {df['file_name'].nunique()}")
lines.append("")
lines.append("sample_file_names:")
for s in sample:
    lines.append(s)

out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB FILE NAME INSPECTION V1 ===")
print(f"unique_file_names: {df['file_name'].nunique()}")
print("\nSample file names:")
for s in sample[:40]:
    print(s)

print(f"\nSaved: {out}")

