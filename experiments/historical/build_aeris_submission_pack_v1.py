from pathlib import Path
import pandas as pd
import zipfile

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"
pack_dir = exp_dir / "AERIS_submission_pack_v1"
pack_dir.mkdir(parents=True, exist_ok=True)

files_to_copy = [
    "AERIS_submission_validation_memo_v1.txt",
    "AERIS_submission_validation_checklist_v1.csv",
    "AERIS_CVB_validation_snapshot_v1.txt",
    "AERIS_CVB_validation_claims_v1.csv",
    "cvb_clip_alert_feed_v1.csv",
    "cvb_clip_alert_feed_v3_conservative.csv",
    "cvb_clip_alert_feed_v4_visibility_gated.csv",
    "cvb_aeris_vs_validated_reference_v1.csv",
    "cvb_aeris_vs_validated_reference_v2_v4.csv",
    "cvb_aeris_vs_validated_reference_v2_v4_summary.txt",
    "cvb_behavior_code_calibration_v1.csv",
    "cvb_real_alert_candidates_v1.csv",
    "cvb_review_pack_v1.csv",
    "cvb_manual_adjudication_template_v1.csv",
    "cvb_top20_clip_ids_only_v1.txt",
]

manifest_rows = []

for name in files_to_copy:
    src = exp_dir / name
    dst = pack_dir / name
    exists = src.exists()
    if exists:
        dst.write_bytes(src.read_bytes())
        size = src.stat().st_size
    else:
        size = 0

    manifest_rows.append({
        "file_name": name,
        "exists": exists,
        "size_bytes": size
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_out = pack_dir / "submission_pack_manifest_v1.csv"
manifest_df.to_csv(manifest_out, index=False)

readme = []
readme.append("AERIS SUBMISSION PACK V1")
readme.append("========================")
readme.append("")
readme.append("Purpose:")
readme.append("- Collect the current real-video validation evidence in one place.")
readme.append("")
readme.append("Important positioning:")
readme.append("- This package supports a validated intermediate benchmark stage on natural video-derived CVB data.")
readme.append("- It does NOT yet support the strongest claim of full final validation.")
readme.append("- Human adjudication remains the next required validation layer.")
readme.append("")
readme.append("Included files:")
for _, row in manifest_df.iterrows():
    readme.append(f"- {row['file_name']} | exists={row['exists']} | size_bytes={row['size_bytes']}")

readme_out = pack_dir / "README_submission_pack_v1.txt"
readme_out.write_text("\n".join(readme), encoding="utf-8")

zip_out = exp_dir / "AERIS_submission_pack_v1.zip"
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in pack_dir.iterdir():
        if p.is_file():
            zf.write(p, arcname=p.name)

print("\n=== AERIS SUBMISSION PACK V1 ===")
print(manifest_df.to_string(index=False))
print(f"\nSaved folder : {pack_dir}")
print(f"Saved zip    : {zip_out}")

