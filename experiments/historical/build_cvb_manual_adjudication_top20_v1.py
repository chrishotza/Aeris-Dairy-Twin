import pandas as pd
from pathlib import Path
import os

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_manual_adjudication_template_v1.csv"
out = exp_dir / "cvb_manual_adjudication_top20_v1.csv"

df = pd.read_csv(inp)
top20 = df.head(20).copy()
top20.to_csv(out, index=False)

print("\n=== CVB MANUAL ADJUDICATION TOP20 V1 ===")
print(f"rows: {len(top20)}")
print(f"Saved: {out}")

