import pandas as pd
import numpy as np
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

beh_in   = exp_dir / "cvb_behavior_table_v1.csv"
aeris_in = exp_dir / "cvb_clip_alert_feed_v3_conservative.csv"

ref_out  = exp_dir / "cvb_validated_reference_by_clip_v1.csv"
cmp_out  = exp_dir / "cvb_aeris_vs_validated_reference_v1.csv"
sum_out  = exp_dir / "cvb_aeris_vs_validated_reference_summary_v1.txt"

# =========================================================
# Load
# =========================================================

beh = pd.read_csv(beh_in)
aeris = pd.read_csv(aeris_in)

beh["file_name"] = beh["file_name"].fillna("missing").astype(str)
beh["behavior"] = beh["behavior"].fillna("missing").astype(str)

def extract_clip_id(path_str: str) -> str:
    try:
        return Path(path_str).parent.name
    except Exception:
        return "unknown_clip"

beh["clip_id"] = beh["file_name"].apply(extract_clip_id)

# =========================================================
# Build validated reference from real behavior labels
# Conservative mapping:
# - running => RED evidence
# - walking / hidden / other / none => caution / ambiguity
# - grazing / resting / ruminating / drinking / grooming => normal observable
# =========================================================

counts = (
    beh.groupby(["clip_id", "behavior"])
       .size()
       .reset_index(name="count")
)

pivot = (
    counts.pivot_table(
        index="clip_id",
        columns="behavior",
        values="count",
        fill_value=0,
        aggfunc="sum"
    )
    .reset_index()
)

expected = [
    "grazing",
    "resting-lying",
    "resting-standing",
    "ruminating-lying",
    "ruminating-standing",
    "drinking",
    "walking",
    "grooming",
    "running",
    "hidden",
    "other",
    "none",
    "missing",
]
for c in expected:
    if c not in pivot.columns:
        pivot[c] = 0

pivot["total_obs"] = pivot[expected].sum(axis=1)

for c in expected:
    pivot[f"p_{c.replace('-', '_')}"] = np.where(
        pivot["total_obs"] > 0,
        pivot[c] / pivot["total_obs"],
        0.0
    )

pivot["p_normal"] = (
    pivot["p_grazing"] +
    pivot["p_resting_lying"] +
    pivot["p_resting_standing"] +
    pivot["p_ruminating_lying"] +
    pivot["p_ruminating_standing"] +
    pivot["p_drinking"] +
    pivot["p_grooming"]
)

pivot["p_ambiguous"] = (
    pivot["p_hidden"] +
    pivot["p_other"] +
    pivot["p_none"]
)

pivot["p_mobile"] = pivot["p_walking"]
pivot["p_extreme"] = pivot["p_running"]

def validated_reference(row):
    p_extreme = row["p_extreme"]
    p_amb = row["p_ambiguous"]
    p_mobile = row["p_mobile"]
    total = row["total_obs"]

    # very little evidence
    if total < 20:
        return ("WATCH", "low_evidence")

    # strong explicit anomaly in validated behavior
    if p_extreme >= 0.08:
        return ("RED", "running_present")

    # weaker but still concerning anomaly
    if p_extreme > 0.0:
        return ("YELLOW", "minor_running_signal")

    # ambiguity / visibility / atypical content
    if p_amb >= 0.20:
        return ("WATCH", "ambiguous_or_hidden_content")

    # lots of walking is not automatically pathological, but merits caution
    if p_mobile >= 0.45:
        return ("WATCH", "mobile_behavior_dominant")

    return ("GREEN", "normal_behavior_dominant")

refs = pivot.apply(validated_reference, axis=1)
pivot["validated_ref_severity"] = [x[0] for x in refs]
pivot["validated_ref_reason"] = [x[1] for x in refs]

pivot.to_csv(ref_out, index=False)

# =========================================================
# Compare against AERIS conservative feed
# =========================================================

cmp = aeris.merge(
    pivot[["clip_id", "validated_ref_severity", "validated_ref_reason", "p_normal", "p_ambiguous", "p_mobile", "p_extreme"]],
    on="clip_id",
    how="left"
)

# exact agreement
cmp["exact_match"] = (cmp["severity_conservative"] == cmp["validated_ref_severity"]).astype(int)

# concern agreement: only YELLOW/RED count as true concern
def concern_flag(x):
    return str(x) in {"YELLOW", "RED"}

cmp["aeris_concern"] = cmp["severity_conservative"].apply(concern_flag).astype(int)
cmp["ref_concern"] = cmp["validated_ref_severity"].apply(concern_flag).astype(int)

tp = int(((cmp["aeris_concern"] == 1) & (cmp["ref_concern"] == 1)).sum())
fp = int(((cmp["aeris_concern"] == 1) & (cmp["ref_concern"] == 0)).sum())
fn = int(((cmp["aeris_concern"] == 0) & (cmp["ref_concern"] == 1)).sum())
tn = int(((cmp["aeris_concern"] == 0) & (cmp["ref_concern"] == 0)).sum())

precision = tp / (tp + fp + 1e-12)
recall = tp / (tp + fn + 1e-12)
specificity = tn / (tn + fp + 1e-12)
accuracy = (tp + tn) / max(len(cmp), 1)

# disagreement shortlist
cmp["disagreement_strength"] = (
    (cmp["severity_conservative"] != cmp["validated_ref_severity"]).astype(int) *
    (cmp["candidate_score"].fillna(0.0) + cmp["p_extreme"].fillna(0.0) + cmp["p_ambiguous"].fillna(0.0))
)

cmp = cmp.sort_values(
    ["disagreement_strength", "candidate_score"],
    ascending=False
).reset_index(drop=True)

cmp.to_csv(cmp_out, index=False)

# =========================================================
# Summary
# =========================================================

lines = []
lines.append("CVB AERIS VS VALIDATED REFERENCE V1")
lines.append("===================================")
lines.append(f"rows_compared: {len(cmp)}")
lines.append("")
lines.append("aeris_conservative_counts:")
for k, v in cmp["severity_conservative"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("validated_reference_counts:")
for k, v in cmp["validated_ref_severity"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append(f"exact_match_rate: {cmp['exact_match'].mean():.4f}")
lines.append(f"concern_precision: {precision:.4f}")
lines.append(f"concern_recall: {recall:.4f}")
lines.append(f"concern_specificity: {specificity:.4f}")
lines.append(f"concern_accuracy: {accuracy:.4f}")
lines.append("")
lines.append("top_disagreements:")
for _, row in cmp[cmp["severity_conservative"] != cmp["validated_ref_severity"]].head(25).iterrows():
    lines.append(
        f"- {row['clip_id']} | AERIS={row['severity_conservative']} | REF={row['validated_ref_severity']} | "
        f"reason={row['validated_ref_reason']} | score={row['candidate_score']:.4f} | "
        f"p_extreme={row['p_extreme']:.4f} | p_ambiguous={row['p_ambiguous']:.4f}"
    )

sum_out.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB AERIS VS VALIDATED REFERENCE V1 ===")
print(f"rows compared       : {len(cmp)}")

print("\nAERIS conservative counts:")
print(cmp["severity_conservative"].value_counts())

print("\nValidated reference counts:")
print(cmp["validated_ref_severity"].value_counts())

print("\nMetrics:")
print(f"exact_match_rate    : {cmp['exact_match'].mean():.4f}")
print(f"concern_precision   : {precision:.4f}")
print(f"concern_recall      : {recall:.4f}")
print(f"concern_specificity : {specificity:.4f}")
print(f"concern_accuracy    : {accuracy:.4f}")

print("\nTop disagreements:")
dis = cmp[cmp["severity_conservative"] != cmp["validated_ref_severity"]].head(25)
if len(dis) == 0:
    print("none")
else:
    print(
        dis[[
            "clip_id","severity_conservative","validated_ref_severity",
            "validated_ref_reason","candidate_score","p_extreme","p_ambiguous","p_mobile"
        ]].to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {ref_out}")
print(f"- {cmp_out}")
print(f"- {sum_out}")

