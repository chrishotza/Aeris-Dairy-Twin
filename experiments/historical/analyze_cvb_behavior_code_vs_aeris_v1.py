import re
import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_aeris_projection_v2.csv"
out_csv = exp_dir / "cvb_behavior_code_vs_aeris_v1.csv"
out_txt = exp_dir / "cvb_behavior_code_vs_aeris_summary_v1.txt"

df = pd.read_csv(inp, dtype={"animal_id": str})
df["clip_id"] = df["clip_id"].astype(str)

def extract_beh_code(clip_id: str):
    m = re.search(r'beh(\d+)', clip_id)
    return m.group(1) if m else "unknown"

df["behavior_code"] = df["clip_id"].apply(extract_beh_code)

cross = pd.crosstab(df["behavior_code"], df["aeris_color"])
cross.to_csv(out_csv)

lines = []
lines.append("CVB BEHAVIOR CODE VS AERIS V1")
lines.append("=============================")
lines.append(f"rows: {len(df)}")
lines.append(f"unique_behavior_codes: {df['behavior_code'].nunique()}")
lines.append("")
lines.append("counts_by_behavior_code_and_color:")
lines.append(cross.to_string())

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB BEHAVIOR CODE VS AERIS V1 ===")
print(f"rows                  : {len(df)}")
print(f"unique behavior codes : {df['behavior_code'].nunique()}")
print("\nCross-tab:")
print(cross)

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

