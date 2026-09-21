import os
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pandas as pd

root = Path(".")
cvb_base = root / "11_real_data" / "cvb_data"
review_csv = cvb_base / "exports" / "cvb_review_pack_v1.csv"
frames_root = cvb_base / "raw_frames"
out_dir = cvb_base / "exports" / "cvb_review_gallery_v1"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(review_csv)

def find_clip_folder(clip_id: str):
    matches = list(frames_root.rglob(clip_id))
    for m in matches:
        if m.is_dir():
            return m
    return None

def make_contact_sheet(folder: Path, out_path: Path, label: str):
    imgs = sorted([p for p in folder.glob("*.jpg")])
    if not imgs:
        return False

    # tomar 6 frames repartidos
    idxs = [0, len(imgs)//5, 2*len(imgs)//5, 3*len(imgs)//5, 4*len(imgs)//5, len(imgs)-1]
    idxs = sorted(set(max(0, min(len(imgs)-1, i)) for i in idxs))

    selected = []
    for i in idxs:
        try:
            im = Image.open(imgs[i]).convert("RGB")
            im = ImageOps.contain(im, (320, 180))
            canvas = Image.new("RGB", (320, 180), "black")
            x = (320 - im.width) // 2
            y = (180 - im.height) // 2
            canvas.paste(im, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 120, 22), fill="black")
            draw.text((6, 4), imgs[i].name, fill="white")
            selected.append(canvas)
        except Exception:
            continue

    if not selected:
        return False

    header_h = 40
    cols = 3
    rows = 2
    sheet = Image.new("RGB", (cols*320, header_h + rows*180), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header_h), fill="black")
    draw.text((10, 10), label, fill="white")

    for n, im in enumerate(selected[:6]):
        x = (n % cols) * 320
        y = header_h + (n // cols) * 180
        sheet.paste(im, (x, y))

    sheet.save(out_path)
    return True

rows = []
for i, row in df.head(20).iterrows():
    clip_id = str(row["clip_id"])
    folder = find_clip_folder(clip_id)
    out_png = out_dir / f"{i+1:02d}_{clip_id}.png"

    found = folder is not None
    built = False
    if found:
        label = f"{clip_id} | sev={row.get('severity_conservative','')} | score={row.get('candidate_score','')}"
        built = make_contact_sheet(folder, out_png, label)

    rows.append({
        "rank": i + 1,
        "clip_id": clip_id,
        "folder_found": found,
        "gallery_created": built,
        "gallery_file": str(out_png) if built else ""
    })

manifest = pd.DataFrame(rows)
manifest_out = cvb_base / "exports" / "cvb_review_gallery_manifest_v1.csv"
manifest.to_csv(manifest_out, index=False)

print("\n=== CVB REVIEW GALLERY V1 ===")
print(manifest.to_string(index=False))
print(f"\nSaved gallery folder: {out_dir}")
print(f"Saved manifest      : {manifest_out}")

