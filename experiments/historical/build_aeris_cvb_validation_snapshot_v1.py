import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

cmp_in = exp_dir / "cvb_aeris_vs_validated_reference_v2_v4.csv"
review_in = exp_dir / "cvb_review_pack_v1.csv"
manual_in = exp_dir / "cvb_manual_adjudication_template_v1.csv"

out_txt = exp_dir / "AERIS_CVB_validation_snapshot_v1.txt"
out_csv = exp_dir / "AERIS_CVB_validation_claims_v1.csv"

df = pd.read_csv(cmp_in)
review = pd.read_csv(review_in)
manual = pd.read_csv(manual_in)

claims = []

claims.append({
    "claim": "AERIS was evaluated on natural video-derived CVB clips",
    "status": "supported",
    "evidence": f"rows_compared={len(df)}"
})

claims.append({
    "claim": "Visibility gating improved agreement vs validated reference",
    "status": "supported",
    "evidence": "exact_match_rate 0.6748 -> 0.7500; specificity 0.8307 -> 0.9065; accuracy 0.8319 -> 0.9071; improved_rows=34; worsened_rows=0"
})

claims.append({
    "claim": "AERIS maintained full concern recall against this validated reference split",
    "status": "supported_with_caution",
    "evidence": "concern_recall=1.0000, but only 3 validated concern positives in this comparison"
})

claims.append({
    "claim": "AERIS is fully validated for presentation as a final proven detector",
    "status": "not_yet_supported",
    "evidence": "manual adjudication of prioritized clips still pending"
})

claims_df = pd.DataFrame(claims)
claims_df.to_csv(out_csv, index=False)

lines = []
lines.append("AERIS CVB VALIDATION SNAPSHOT V1")
lines.append("================================")
lines.append(f"rows_compared: {len(df)}")
lines.append("")
lines.append("Current validated-reference metrics (V4):")
lines.append("- exact_match_rate: 0.7500")
lines.append("- concern_precision: 0.0667")
lines.append("- concern_recall: 1.0000")
lines.append("- concern_specificity: 0.9065")
lines.append("- concern_accuracy: 0.9071")
lines.append("")
lines.append("Interpretation:")
lines.append("- visibility/ambiguity gating clearly helped")
lines.append("- false concerns dropped materially")
lines.append("- recall remained intact on this split")
lines.append("- precision is still weak because the validated concern class is tiny")
lines.append("")
lines.append(f"review_pack_rows: {len(review)}")
lines.append(f"manual_adjudication_rows: {len(manual)}")
lines.append("")
lines.append("What is already defendable:")
lines.append("- AERIS works on natural video-derived behavioral annotations")
lines.append("- AERIS can produce clip-level alerts on real data")
lines.append("- AERIS improves after ambiguity-aware gating")
lines.append("")
lines.append("What is still needed for strong final validation:")
lines.append("- fill the human adjudication template on the 50 prioritized clips")
lines.append("- compare AERIS vs human judgment")
lines.append("- report false positives, false negatives, and ambiguity-driven errors")
lines.append("")
lines.append("Claim matrix:")
for _, row in claims_df.iterrows():
    lines.append(f"- {row['claim']} | {row['status']} | {row['evidence']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== AERIS CVB VALIDATION SNAPSHOT V1 ===")
print(f"rows_compared: {len(df)}")
print(f"review_pack_rows: {len(review)}")
print(f"manual_adjudication_rows: {len(manual)}")
print("\nClaims:")
print(claims_df.to_string(index=False))
print(f"\nSaved:")
print(f"- {out_txt}")
print(f"- {out_csv}")

