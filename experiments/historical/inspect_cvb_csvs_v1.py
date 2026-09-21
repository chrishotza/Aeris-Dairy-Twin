from pathlib import Path
import pandas as pd

root = Path(".")
base = root / "11_real_data" / "cvb_data" / "data"
exp_dir = root / "11_real_data" / "cvb_data" / "exports"
exp_dir.mkdir(parents=True, exist_ok=True)

csv_files = list(base.rglob("*.csv"))

summary_rows = []
preview_txt = []

for cf in csv_files:
    try:
        df = pd.read_csv(cf)
        cols = list(df.columns)
        summary_rows.append({
            "file_name": cf.name,
            "relative_path": str(cf.relative_to(base)),
            "n_rows": len(df),
            "n_cols": len(cols),
            "columns": ";".join(map(str, cols))
        })

        preview_txt.append(f"=== FILE: {cf.relative_to(base)} ===")
        preview_txt.append(f"rows={len(df)} cols={len(cols)}")
        preview_txt.append("columns: " + ", ".join(map(str, cols)))
        preview_txt.append(df.head(10).to_string(index=False))
        preview_txt.append("")

    except Exception as e:
        summary_rows.append({
            "file_name": cf.name,
            "relative_path": str(cf.relative_to(base)),
            "n_rows": None,
            "n_cols": None,
            "columns": f"ERROR: {e}"
        })

summary_df = pd.DataFrame(summary_rows).sort_values("relative_path").reset_index(drop=True)

summary_out = exp_dir / "cvb_csv_inventory_v1.csv"
preview_out = exp_dir / "cvb_csv_preview_v1.txt"

summary_df.to_csv(summary_out, index=False)
preview_out.write_text("\n".join(preview_txt), encoding="utf-8")

print("\n=== CVB CSV INSPECTION V1 ===")
print(summary_df.to_string(index=False))
print(f"\nSaved:")
print(f"- {summary_out}")
print(f"- {preview_out}")

