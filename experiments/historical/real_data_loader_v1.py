import os
from pathlib import Path
import pandas as pd

root = Path(".")
real_dir = root / "11_real_data"
video_dir = real_dir / "videos"
frames_dir = real_dir / "frames"
exports_dir = real_dir / "exports"

for d in [real_dir, video_dir, frames_dir, exports_dir]:
    d.mkdir(parents=True, exist_ok=True)

video_exts = {".mp4", ".avi", ".mov", ".mkv"}
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

video_rows = []
frame_rows = []

for p in video_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in video_exts:
        video_rows.append({
            "file_name": p.name,
            "full_path": str(p.resolve()),
            "extension": p.suffix.lower(),
            "file_size_bytes": p.stat().st_size
        })

for p in frames_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in image_exts:
        frame_rows.append({
            "file_name": p.name,
            "full_path": str(p.resolve()),
            "extension": p.suffix.lower(),
            "file_size_bytes": p.stat().st_size,
            "parent_folder": p.parent.name
        })

videos_df = pd.DataFrame(video_rows)
frames_df = pd.DataFrame(frame_rows)

videos_out = exports_dir / "real_video_manifest_v1.csv"
frames_out = exports_dir / "real_frame_manifest_v1.csv"
summary_out = exports_dir / "real_data_summary_v1.txt"

videos_df.to_csv(videos_out, index=False)
frames_df.to_csv(frames_out, index=False)

lines = []
lines.append("REAL DATA SUMMARY V1")
lines.append("====================")
lines.append(f"video_files: {len(videos_df)}")
lines.append(f"frame_files: {len(frames_df)}")
lines.append("")
lines.append("folders:")
lines.append(f"- videos: {video_dir}")
lines.append(f"- frames: {frames_dir}")
lines.append(f"- exports: {exports_dir}")
lines.append("")
lines.append("next_step:")
lines.append("- place public dataset files into 11_real_data/videos or 11_real_data/frames")
lines.append("- rerun this script to build manifests")
lines.append("- then build feature extraction on top of the manifest")

summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== REAL DATA LOADER V1 ===")
print(f"videos found : {len(videos_df)}")
print(f"frames found : {len(frames_df)}")
print(f"\nSaved:")
print(f"- {videos_out}")
print(f"- {frames_out}")
print(f"- {summary_out}")
