from pathlib import Path
import pandas as pd

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

memo_out = exp_dir / "AERIS_submission_validation_memo_v1.txt"
check_out = exp_dir / "AERIS_submission_validation_checklist_v1.csv"

memo = []
memo.append("AERIS SUBMISSION VALIDATION MEMO V1")
memo.append("===================================")
memo.append("")
memo.append("Current status:")
memo.append("- AERIS was evaluated on natural video-derived CVB clips.")
memo.append("- AERIS generated clip-level alerts on real annotated material.")
memo.append("- Visibility/ambiguity gating materially improved alignment with the validated reference.")
memo.append("")
memo.append("Current validated-reference metrics (V4):")
memo.append("- exact_match_rate: 0.7500")
memo.append("- concern_precision: 0.0667")
memo.append("- concern_recall: 1.0000")
memo.append("- concern_specificity: 0.9065")
memo.append("- concern_accuracy: 0.9071")
memo.append("")
memo.append("Interpretation:")
memo.append("- The system clearly improved after ambiguity-aware gating.")
memo.append("- False concerns were reduced without harming recall on this split.")
memo.append("- Precision remains limited because the validated concern class is very small.")
memo.append("")
memo.append("What is defendable now:")
memo.append("- natural-video-derived evaluation exists")
memo.append("- clip-level real-data alerting exists")
memo.append("- ambiguity handling improves validation performance")
memo.append("")
memo.append("What is NOT yet complete:")
memo.append("- full visual human adjudication on raw frames")
memo.append("- final human-vs-AERIS agreement study")
memo.append("- strong final claim of complete validation")
memo.append("")
memo.append("Main blocker:")
memo.append("- raw_frames for shortlisted clips are not locally available")
memo.append("")
memo.append("Recommended submission wording:")
memo.append("- present this as a validated intermediate real-video benchmark stage, not as final exhaustive proof")
memo.append("- explicitly state that human adjudication on shortlisted clips is the next validation layer")

memo_out.write_text("\n".join(memo), encoding="utf-8")

check = pd.DataFrame([
    ["Natural video-derived evaluation completed", "yes"],
    ["Real clip-level alert feed completed", "yes"],
    ["Visibility-aware recalibration completed", "yes"],
    ["Validated-reference comparison completed", "yes"],
    ["Manual shortlist created", "yes"],
    ["Human adjudication actually filled", "no"],
    ["Raw frames available for visual review", "no"],
    ["Final strong validation claim supportable", "not yet"],
], columns=["item", "status"])

check.to_csv(check_out, index=False)

print("\n=== AERIS SUBMISSION VALIDATION MEMO V1 ===")
print(f"Saved: {memo_out}")
print(f"Saved: {check_out}")

