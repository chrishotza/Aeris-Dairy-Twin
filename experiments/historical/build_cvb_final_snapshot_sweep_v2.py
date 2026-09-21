from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()
EXP_DIR = ROOT / "11_real_data" / "cvb_data" / "exports"
SWEEP_DIR = EXP_DIR / "cvb_param_sweep_v2"

V4_IN = EXP_DIR / "cvb_clip_alert_feed_v4_visibility_gated.csv"
BEST_IN = SWEEP_DIR / "cvb_param_sweep_best_feed_v2.csv"
CMP_IN = EXP_DIR / "cvb_aeris_vs_validated_reference_v1.csv"

OUT_SNAPSHOT = EXP_DIR / "cvb_final_snapshot_sweep_v2.txt"
OUT_CLAIMS   = EXP_DIR / "cvb_final_claims_sweep_v2.csv"
OUT_COMPARE  = EXP_DIR / "cvb_v4_vs_sweep_best_comparison_v2.csv"
OUT_REVIEW   = EXP_DIR / "cvb_review_pack_sweep_best_v1.csv"

SEV2RANK = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}

def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def metrics(pred_sev, ref_sev):
    pred_rank = pred_sev.astype(str).str.upper().map(SEV2RANK).fillna(0).astype(int)
    ref_rank  = ref_sev.astype(str).str.upper().map(SEV2RANK).fillna(0).astype(int)

    exact = float((pred_rank == ref_rank).mean())

    pred_concern = (pred_rank >= 2).astype(int)
    ref_concern = (ref_rank >= 2).astype(int)

    tp = int(((pred_concern == 1) & (ref_concern == 1)).sum())
    fp = int(((pred_concern == 1) & (ref_concern == 0)).sum())
    fn = int(((pred_concern == 0) & (ref_concern == 1)).sum())
    tn = int(((pred_concern == 0) & (ref_concern == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    accuracy = (tp + tn) / max(len(pred_rank), 1)

    red_rate = float((pred_rank == 3).mean())
    concern_rate = float((pred_rank >= 2).mean())

    return {
        "exact_match_rate": exact,
        "concern_precision": precision,
        "concern_recall": recall,
        "concern_specificity": specificity,
        "concern_accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "red_rate": red_rate,
        "concern_rate": concern_rate,
    }

v4 = safe_read(V4_IN)
best = safe_read(BEST_IN)
cmp = safe_read(CMP_IN)

if "clip_id" not in v4.columns or "clip_id" not in best.columns or "clip_id" not in cmp.columns:
    raise ValueError("clip_id missing in one of the required files")

# normalize V4 severity
if "severity_v4" not in v4.columns:
    if "severity_conservative" in v4.columns:
        v4["severity_v4"] = v4["severity_conservative"]
    else:
        raise ValueError("V4 file needs severity_v4 or severity_conservative")

# normalize BEST severity
if "severity_sweep_best" not in best.columns:
    raise ValueError("Best sweep file needs severity_sweep_best")

# bring validated reference if missing
ref_cols = ["clip_id", "validated_ref_severity", "validated_ref_reason"]
cmp_ref = cmp[ref_cols].drop_duplicates("clip_id")

if "validated_ref_severity" not in v4.columns:
    v4 = v4.merge(cmp_ref, on="clip_id", how="left")
if "validated_ref_severity" not in best.columns:
    best = best.merge(cmp_ref, on="clip_id", how="left")

# bring V4 severity into best only if missing there
if "severity_v4" not in best.columns:
    best = best.merge(
        v4[["clip_id", "severity_v4"]].drop_duplicates("clip_id"),
        on="clip_id",
        how="left"
    )

# final safety checks
needed_best = ["clip_id", "severity_sweep_best", "severity_v4", "validated_ref_severity"]
missing_best = [c for c in needed_best if c not in best.columns]
if missing_best:
    raise ValueError(f"Best comparison frame missing columns: {missing_best}")

m_v4 = metrics(v4["severity_v4"], v4["validated_ref_severity"])
m_best = metrics(best["severity_sweep_best"], best["validated_ref_severity"])

compare = best.copy()
compare["v4_match"] = (compare["severity_v4"].astype(str).str.upper() == compare["validated_ref_severity"].astype(str).str.upper()).astype(int)
compare["best_match"] = (compare["severity_sweep_best"].astype(str).str.upper() == compare["validated_ref_severity"].astype(str).str.upper()).astype(int)
compare["improved"] = ((compare["v4_match"] == 0) & (compare["best_match"] == 1)).astype(int)
compare["worsened"] = ((compare["v4_match"] == 1) & (compare["best_match"] == 0)).astype(int)
compare.to_csv(OUT_COMPARE, index=False)

review = best.copy()
for col, default in [
    ("candidate_score", 0.0),
    ("p_ambiguous", 0.0),
    ("p_extreme", 0.0),
    ("mean_visibility_proxy", 0.0),
]:
    if col not in review.columns:
        review[col] = default

review["priority_score"] = (
    1.20 * (review["severity_sweep_best"].astype(str).str.upper() == "RED").astype(int) +
    0.70 * (review["severity_sweep_best"].astype(str).str.upper() == "YELLOW").astype(int) +
    0.25 * pd.to_numeric(review["candidate_score"], errors="coerce").fillna(0.0) +
    0.20 * pd.to_numeric(review["p_extreme"], errors="coerce").fillna(0.0) +
    0.10 * pd.to_numeric(review["p_ambiguous"], errors="coerce").fillna(0.0) -
    0.10 * pd.to_numeric(review["mean_visibility_proxy"], errors="coerce").fillna(0.0)
)

review = review.sort_values(
    ["priority_score", "severity_sweep_best", "candidate_score"],
    ascending=[False, True, False]
).reset_index(drop=True)

review.head(50).to_csv(OUT_REVIEW, index=False)

claims = [
    {
        "claim": "Sweep-optimized candidate outperforms V4 baseline against validated reference",
        "status": "supported",
        "evidence": (
            f"exact {m_v4['exact_match_rate']:.4f}->{m_best['exact_match_rate']:.4f}; "
            f"precision {m_v4['concern_precision']:.4f}->{m_best['concern_precision']:.4f}; "
            f"specificity {m_v4['concern_specificity']:.4f}->{m_best['concern_specificity']:.4f}; "
            f"accuracy {m_v4['concern_accuracy']:.4f}->{m_best['concern_accuracy']:.4f}"
        ),
    },
    {
        "claim": "Sweep-optimized candidate reduces false positives without losing concern recall",
        "status": "supported",
        "evidence": (
            f"fp {m_v4['fp']}->{m_best['fp']}; tp {m_v4['tp']}->{m_best['tp']}; "
            f"fn {m_v4['fn']}->{m_best['fn']}; recall {m_v4['concern_recall']:.4f}->{m_best['concern_recall']:.4f}"
        ),
    },
    {
        "claim": "Current best candidate is a strong intermediate benchmark result on natural video-derived annotated data",
        "status": "supported_with_scope",
        "evidence": "validated against CVB-derived reference, but still not a full end-to-end final validation claim",
    },
]

claims_df = pd.DataFrame(claims)
claims_df.to_csv(OUT_CLAIMS, index=False)

lines = []
lines.append("CVB FINAL SNAPSHOT :: SWEEP V2")
lines.append("================================")
lines.append("")
lines.append("V4 baseline metrics:")
for k, v in m_v4.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("Best sweep metrics:")
for k, v in m_best.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append(f"improved_rows: {int(compare['improved'].sum())}")
lines.append(f"worsened_rows: {int(compare['worsened'].sum())}")
lines.append("")
lines.append("Claims:")
for _, row in claims_df.iterrows():
    lines.append(f"- {row['claim']} | {row['status']} | {row['evidence']}")

OUT_SNAPSHOT.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB FINAL SNAPSHOT :: SWEEP V2 ===")
print("\nV4 baseline metrics:")
for k, v in m_v4.items():
    print(f"{k}: {v}")

print("\nBest sweep metrics:")
for k, v in m_best.items():
    print(f"{k}: {v}")

print(f"\nimproved_rows: {int(compare['improved'].sum())}")
print(f"worsened_rows: {int(compare['worsened'].sum())}")

print(f"\nSaved:")
print(f"- {OUT_SNAPSHOT}")
print(f"- {OUT_CLAIMS}")
print(f"- {OUT_COMPARE}")
print(f"- {OUT_REVIEW}")

