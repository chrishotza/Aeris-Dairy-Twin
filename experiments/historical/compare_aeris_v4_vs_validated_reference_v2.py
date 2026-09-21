import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

cmp_v1_in = exp_dir / "cvb_aeris_vs_validated_reference_v1.csv"
v4_in     = exp_dir / "cvb_clip_alert_feed_v4_visibility_gated.csv"

out_csv   = exp_dir / "cvb_aeris_vs_validated_reference_v2_v4.csv"
out_txt   = exp_dir / "cvb_aeris_vs_validated_reference_v2_v4_summary.txt"

base = pd.read_csv(cmp_v1_in)
v4   = pd.read_csv(v4_in)

df = base.merge(
    v4[["clip_id", "severity_v4", "recommended_action_v4", "priority_v4"]],
    on="clip_id",
    how="left"
)

def concern_flag(x):
    return str(x) in {"YELLOW", "RED"}

def metrics_for(col_name):
    exact = (df[col_name] == df["validated_ref_severity"]).mean()

    aeris_concern = df[col_name].apply(concern_flag).astype(int)
    ref_concern = df["validated_ref_severity"].apply(concern_flag).astype(int)

    tp = int(((aeris_concern == 1) & (ref_concern == 1)).sum())
    fp = int(((aeris_concern == 1) & (ref_concern == 0)).sum())
    fn = int(((aeris_concern == 0) & (ref_concern == 1)).sum())
    tn = int(((aeris_concern == 0) & (ref_concern == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    accuracy = (tp + tn) / max(len(df), 1)

    return {
        "exact_match_rate": exact,
        "concern_precision": precision,
        "concern_recall": recall,
        "concern_specificity": specificity,
        "concern_accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

m_v3 = metrics_for("severity_conservative")
m_v4 = metrics_for("severity_v4")

# rows that improved from disagreement to agreement
df["v3_match"] = (df["severity_conservative"] == df["validated_ref_severity"]).astype(int)
df["v4_match"] = (df["severity_v4"] == df["validated_ref_severity"]).astype(int)

improved = df[(df["v3_match"] == 0) & (df["v4_match"] == 1)].copy()
worsened = df[(df["v3_match"] == 1) & (df["v4_match"] == 0)].copy()

df.to_csv(out_csv, index=False)

lines = []
lines.append("CVB AERIS VS VALIDATED REFERENCE V2 (V4 GATED)")
lines.append("==============================================")
lines.append(f"rows: {len(df)}")
lines.append("")
lines.append("V3 metrics:")
for k, v in m_v3.items():
    lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
lines.append("")
lines.append("V4 metrics:")
for k, v in m_v4.items():
    lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
lines.append("")
lines.append(f"improved_rows: {len(improved)}")
lines.append(f"worsened_rows: {len(worsened)}")
lines.append("")
lines.append("top_improved_rows:")
for _, row in improved.head(25).iterrows():
    lines.append(
        f"- {row['clip_id']} | V3={row['severity_conservative']} | V4={row['severity_v4']} | "
        f"REF={row['validated_ref_severity']} | reason={row['validated_ref_reason']}"
    )
lines.append("")
lines.append("top_worsened_rows:")
for _, row in worsened.head(25).iterrows():
    lines.append(
        f"- {row['clip_id']} | V3={row['severity_conservative']} | V4={row['severity_v4']} | "
        f"REF={row['validated_ref_severity']} | reason={row['validated_ref_reason']}"
    )

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB AERIS VS VALIDATED REFERENCE V2 (V4 GATED) ===")
print(f"rows: {len(df)}")

print("\nV3 metrics:")
for k, v in m_v3.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\nV4 metrics:")
for k, v in m_v4.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print(f"\nimproved_rows: {len(improved)}")
print(f"worsened_rows: {len(worsened)}")

print("\nTop improved rows:")
if len(improved) == 0:
    print("none")
else:
    print(
        improved[[
            "clip_id","severity_conservative","severity_v4",
            "validated_ref_severity","validated_ref_reason"
        ]].head(25).to_string(index=False)
    )

print("\nTop worsened rows:")
if len(worsened) == 0:
    print("none")
else:
    print(
        worsened[[
            "clip_id","severity_conservative","severity_v4",
            "validated_ref_severity","validated_ref_reason"
        ]].head(25).to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

