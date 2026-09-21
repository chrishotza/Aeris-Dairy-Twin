from pathlib import Path
import pandas as pd

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_top20_clip_ids_only_v1.txt"
out = exp_dir / "cvb_rclone_include_top20_v1.txt"

clip_ids = [x.strip() for x in Path(inp).read_text(encoding="utf-8").splitlines() if x.strip()]

lines = []
for clip_id in clip_ids:
    lines.append(f"+ */{clip_id}/**")
lines.append("- **")

Path(out).write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB RCLONE INCLUDE TOP20 V1 ===")
for x in lines[:10]:
    print(x)
print("...")
print(f"\nSaved: {out}")

