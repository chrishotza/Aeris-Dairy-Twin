import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_clip_alert_feed_v3_conservative.csv"
cmp_in = exp_dir / "cvb_aeris_vs_validated_reference_v1.csv"

out_csv = exp_dir / "cvb_clip_alert_feed_v4_visibility_gated.csv"
out_txt = exp_dir / "cvb_clip_alert_feed_v4_visibility_gated_summary.txt"

df = pd.read_csv(inp)
cmp = pd.read_csv(cmp_in)

# bring in validated-reference-side ambiguity signals
merge_cols = ["clip_id", "p_ambiguous", "p_extreme", "validated_ref_severity", "validated_ref_reason"]
df = df.merge(cmp[merge_cols].drop_duplicates("clip_id"), on="clip_id", how="left")

severity_order = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
inv_severity = {v: k for k, v in severity_order.items()}

def gate_visibility(row):
    sev = row["severity_conservative"]
    rank = severity_order.get(sev, 0)

    p_amb = float(row.get("p_ambiguous", 0.0) or 0.0)
    p_ext = float(row.get("p_extreme", 0.0) or 0.0)
    vis = float(row.get("mean_visibility_proxy", 0.0) or 0.0)

    # hard rule:
    # if ambiguity is high and there is no explicit extreme behavior,
    # cap severity to WATCH
    if p_ext == 0.0 and (p_amb >= 0.35 or vis < 0.55):
        rank = min(rank, severity_order["WATCH"])

    # if ambiguity is very high, never let it be RED without extreme signal
    if p_ext == 0.0 and p_amb >= 0.25 and rank == severity_order["RED"]:
        rank = severity_order["WATCH"]

    # only explicit extreme signal can preserve RED safely
    new_sev = inv_severity[rank]

    if new_sev == "RED":
        action = "inspect_immediately"
        priority = "critical"
    elif new_sev == "YELLOW":
        action = "review_context_and_monitor"
        priority = "high"
    elif new_sev == "WATCH":
        action = "collect_more_visual_evidence"
        priority = "medium"
    else:
        action = "standard_monitoring"
        priority = "low"

    return pd.Series([new_sev, action, priority])

df[["severity_v4", "recommended_action_v4", "priority_v4"]] = df.apply(gate_visibility, axis=1)

changed = df[df["severity_conservative"] != df["severity_v4"]].copy()

df.to_csv(out_csv, index=False)

lines = []
lines.append("CVB CLIP ALERT FEED V4 VISIBILITY GATED")
lines.append("=======================================")
lines.append(f"rows: {len(df)}")
lines.append("")
lines.append("v3_counts:")
for k, v in df["severity_conservative"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("v4_counts:")
for k, v in df["severity_v4"].value_counts().to_dict().items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append(f"changed_rows: {len(changed)}")
lines.append("")
lines.append("top_changed_rows:")
for _, row in changed.head(30).iterrows():
    lines.append(
        f"- {row['clip_id']} | {row['severity_conservative']} -> {row['severity_v4']} | "
        f"p_ambiguous={row['p_ambiguous']:.4f} | p_extreme={row['p_extreme']:.4f} | "
        f"vis={row['mean_visibility_proxy']:.4f}"
    )

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB CLIP ALERT FEED V4 VISIBILITY GATED ===")
print(f"rows         : {len(df)}")
print(f"changed rows : {len(changed)}")

print("\nV3 counts:")
print(df["severity_conservative"].value_counts())

print("\nV4 counts:")
print(df["severity_v4"].value_counts())

print("\nTop changed rows:")
if len(changed) == 0:
    print("none")
else:
    print(
        changed[[
            "clip_id","severity_conservative","severity_v4",
            "p_ambiguous","p_extreme","mean_visibility_proxy","validated_ref_reason"
        ]]
        .head(30)
        .to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

