from pathlib import Path
import pandas as pd
import html
import zipfile

ROOT = Path(".").resolve()
EXP_DIR = ROOT / "11_real_data" / "cvb_data" / "exports"
SWEEP_DIR = EXP_DIR / "cvb_param_sweep_v2"

SNAPSHOT_IN = EXP_DIR / "cvb_final_snapshot_sweep_v2.txt"
CLAIMS_IN   = EXP_DIR / "cvb_final_claims_sweep_v2.csv"
COMPARE_IN  = EXP_DIR / "cvb_v4_vs_sweep_best_comparison_v2.csv"
REVIEW_IN   = EXP_DIR / "cvb_review_pack_sweep_best_v1.csv"
BEST_FEED_IN= SWEEP_DIR / "cvb_param_sweep_best_feed_v2.csv"
BEST_PAR_IN = SWEEP_DIR / "cvb_param_sweep_best_params_v2.json"
BEST_SUM_IN = SWEEP_DIR / "cvb_param_sweep_summary_v2.txt"

GALLERY_MANIFEST_IN = EXP_DIR / "cvb_review_gallery_manifest_v1.csv"
OLD_ADJ_IN = EXP_DIR / "cvb_manual_adjudication_template_v1.csv"

CHAMP_DIR = EXP_DIR / "AERIS_champion_pack_v1"
CHAMP_DIR.mkdir(parents=True, exist_ok=True)

def must(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path

for p in [SNAPSHOT_IN, CLAIMS_IN, COMPARE_IN, REVIEW_IN, BEST_FEED_IN, BEST_PAR_IN, BEST_SUM_IN]:
    must(p)

review = pd.read_csv(REVIEW_IN)
best = pd.read_csv(BEST_FEED_IN)

gallery_manifest = pd.read_csv(GALLERY_MANIFEST_IN) if GALLERY_MANIFEST_IN.exists() else pd.DataFrame(columns=["clip_id","gallery_created","gallery_file"])
old_adj = pd.read_csv(OLD_ADJ_IN) if OLD_ADJ_IN.exists() else pd.DataFrame(columns=["clip_id"])

# build adjudication template for sweep-best review pack
adj = review.copy()

for c, default in [
    ("reviewed", ""),
    ("human_label", ""),
    ("human_severity", ""),
    ("visibility_ok", ""),
    ("true_concern", ""),
    ("notes", ""),
]:
    if c not in adj.columns:
        adj[c] = default

if len(gallery_manifest):
    adj = adj.merge(
        gallery_manifest[["clip_id", "gallery_created", "gallery_file"]],
        on="clip_id",
        how="left"
    )
else:
    adj["gallery_created"] = False
    adj["gallery_file"] = ""

adj_out = CHAMP_DIR / "cvb_manual_adjudication_sweep_best_v1.csv"
adj.to_csv(adj_out, index=False)

# build html review index from available galleries
gallery_map = {}
if len(gallery_manifest):
    gallery_map = {
        str(r["clip_id"]): str(r["gallery_file"])
        for _, r in gallery_manifest.iterrows()
        if bool(r.get("gallery_created", False))
    }

rows = []
for _, row in review.head(50).iterrows():
    clip_id = str(row["clip_id"])
    sev = row.get("severity_sweep_best", row.get("severity_conservative", ""))
    score = row.get("candidate_score", "")
    ref = row.get("validated_ref_severity", "")
    reason = row.get("validated_ref_reason", "")
    img_path = gallery_map.get(clip_id, "")
    img_rel = Path(img_path).name if img_path else ""

    img_tag = f'<img src="../cvb_review_gallery_v1/{html.escape(img_rel)}" style="max-width:100%; border:1px solid #ccc;">' if img_rel else "<div style='padding:8px;border:1px dashed #999;'>gallery not available for this clip</div>"

    rows.append(f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:16px; margin:16px 0; font-family:Arial,sans-serif;">
      <h3 style="margin:0 0 8px 0;">{html.escape(clip_id)}</h3>
      <div style="margin-bottom:8px;">
        <b>sweep severity:</b> {html.escape(str(sev))} |
        <b>score:</b> {html.escape(str(score))} |
        <b>validated ref:</b> {html.escape(str(ref))} |
        <b>reason:</b> {html.escape(str(reason))}
      </div>
      {img_tag}
    </div>
    """)

html_out = CHAMP_DIR / "cvb_review_index_sweep_best_v1.html"
html_text = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AERIS Sweep-Best Review Index</title>
</head>
<body style="max-width:1100px; margin:24px auto; font-family:Arial,sans-serif;">
  <h1>AERIS Sweep-Best Review Index</h1>
  <p>Champion candidate review page. Fill the CSV:
  <code>cvb_manual_adjudication_sweep_best_v1.csv</code></p>
  <p>Suggested fields: reviewed=yes, human_label=normal/ambiguous/concerning, human_severity=GREEN/WATCH/YELLOW/RED, visibility_ok=yes/no, true_concern=yes/no.</p>
  {''.join(rows)}
</body>
</html>
"""
html_out.write_text(html_text, encoding="utf-8")

# champion memo
memo_out = CHAMP_DIR / "AERIS_champion_memo_v1.txt"
memo = []
memo.append("AERIS CHAMPION MEMO V1")
memo.append("======================")
memo.append("")
memo.append("Current champion:")
memo.append("- cvb_param_sweep_best_feed_v2.csv")
memo.append("")
memo.append("Why champion:")
memo.append("- keeps tp=3 and fn=0")
memo.append("- reduces fp from 42 to 8")
memo.append("- improves exact match, precision, specificity, and accuracy over V4")
memo.append("")
memo.append("Key outputs in this pack:")
memo.append("- final snapshot")
memo.append("- final claims")
memo.append("- v4 vs sweep comparison")
memo.append("- best feed")
memo.append("- best params")
memo.append("- best review pack")
memo.append("- adjudication sheet with gallery links when available")
memo_out.write_text("\n".join(memo), encoding="utf-8")

# copy important files into champion pack
copy_map = [
    SNAPSHOT_IN,
    CLAIMS_IN,
    COMPARE_IN,
    REVIEW_IN,
    BEST_FEED_IN,
    BEST_PAR_IN,
    BEST_SUM_IN,
    adj_out,
    html_out,
    memo_out,
]

manifest_rows = []
for src in copy_map:
    dst = CHAMP_DIR / src.name
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    manifest_rows.append({
        "file_name": dst.name,
        "size_bytes": dst.stat().st_size,
        "source": str(src)
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_out = CHAMP_DIR / "champion_pack_manifest_v1.csv"
manifest_df.to_csv(manifest_out, index=False)

zip_out = EXP_DIR / "AERIS_champion_pack_v1.zip"
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in CHAMP_DIR.iterdir():
        if p.is_file():
            zf.write(p, arcname=p.name)

print("\n=== AERIS CHAMPION PACK V1 ===")
print(manifest_df.to_string(index=False))
print(f"\nSaved folder : {CHAMP_DIR}")
print(f"Saved zip    : {zip_out}")
print(f"Saved html   : {html_out}")
print(f"Saved adj    : {adj_out}")

