from pathlib import Path
import pandas as pd
import html

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

review_pack = pd.read_csv(exp_dir / "cvb_review_pack_v1.csv")
gallery_manifest = pd.read_csv(exp_dir / "cvb_review_gallery_manifest_v1.csv")
adj = pd.read_csv(exp_dir / "cvb_manual_adjudication_template_v1.csv")

# merge gallery paths into adjudication sheet
merged = adj.merge(
    gallery_manifest[["clip_id", "gallery_created", "gallery_file"]],
    on="clip_id",
    how="left"
)

out_csv = exp_dir / "cvb_manual_adjudication_with_gallery_v1.csv"
merged.to_csv(out_csv, index=False)

# build html review index
gallery_map = {
    str(row["clip_id"]): str(row["gallery_file"])
    for _, row in gallery_manifest.iterrows()
    if bool(row["gallery_created"])
}

rows = []
for _, row in review_pack.iterrows():
    clip_id = str(row["clip_id"])
    img_path = gallery_map.get(clip_id, "")
    img_rel = Path(img_path).name if img_path else ""
    severity = row.get("severity_conservative", "")
    score = row.get("candidate_score", "")
    n_red = row.get("n_red", "")
    n_yellow = row.get("n_yellow", "")
    n_hold = row.get("n_hold", "")
    action = row.get("recommended_action_conservative", row.get("recommended_action", ""))

    img_tag = f'<img src="cvb_review_gallery_v1/{html.escape(img_rel)}" style="max-width:100%; border:1px solid #ccc;">' if img_rel else "<div>missing image</div>"

    rows.append(f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:16px; margin:16px 0; font-family:Arial,sans-serif;">
      <h3 style="margin:0 0 8px 0;">{html.escape(clip_id)}</h3>
      <div style="margin-bottom:8px;">
        <b>Severity:</b> {html.escape(str(severity))} |
        <b>Score:</b> {html.escape(str(score))} |
        <b>n_red:</b> {html.escape(str(n_red))} |
        <b>n_yellow:</b> {html.escape(str(n_yellow))} |
        <b>n_hold:</b> {html.escape(str(n_hold))} |
        <b>Action:</b> {html.escape(str(action))}
      </div>
      {img_tag}
    </div>
    """)

html_text = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CVB Review Index V1</title>
</head>
<body style="max-width:1100px; margin:24px auto; font-family:Arial,sans-serif;">
  <h1>CVB Review Index V1</h1>
  <p>Use this page to inspect the gallery PNGs, and fill the CSV:
  <code>cvb_manual_adjudication_with_gallery_v1.csv</code></p>
  <p>Recommended fields:
  reviewed=yes, human_label=normal/ambiguous/concerning, human_severity=GREEN/WATCH/YELLOW/RED, visibility_ok=yes/no, true_concern=yes/no.</p>
  {''.join(rows)}
</body>
</html>
"""

out_html = exp_dir / "cvb_review_index_v1.html"
out_html.write_text(html_text, encoding="utf-8")

print("\n=== CVB REVIEW INDEX V1 ===")
print(f"Saved adjudication csv: {out_csv}")
print(f"Saved html index      : {out_html}")

