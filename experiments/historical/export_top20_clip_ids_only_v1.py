import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_review_pack_v1.csv"
out = exp_dir / "cvb_top20_clip_ids_only_v1.txt"

df = pd.read_csv(inp)
clip_ids = df["clip_id"].dropna().astype(str).head(20).tolist()

Path(out).write_text("\n".join(clip_ids), encoding="utf-8")

print("\n=== TOP20 CLIP IDS ONLY V1 ===")
for x in clip_ids:
    print(x)
print(f"\nSaved: {out}")

