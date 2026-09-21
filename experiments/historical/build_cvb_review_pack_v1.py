import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

inp = exp_dir / "cvb_clip_alert_feed_v3_conservative.csv"
out_csv = exp_dir / "cvb_review_pack_v1.csv"
out_txt = exp_dir / "cvb_review_pack_v1.txt"

df = pd.read_csv(inp)

red = df[df["severity_conservative"] == "RED"].sort_values(
    ["candidate_score", "n_red", "n_yellow"], ascending=False
).head(20)

yellow = df[df["severity_conservative"] == "YELLOW"].sort_values(
    ["candidate_score", "n_red", "n_yellow"], ascending=False
).head(15)

watch = df[df["severity_conservative"] == "WATCH"].sort_values(
    ["candidate_score", "n_red", "n_yellow"], ascending=False
).head(15)

pack = pd.concat([red, yellow, watch], ignore_index=True)
pack.to_csv(out_csv, index=False)

lines = []
lines.append("CVB REVIEW PACK V1")
lines.append("==================")
lines.append(f"rows: {len(pack)}")
lines.append("")
for sev in ["RED", "YELLOW", "WATCH"]:
    lines.append(f"{sev}:")
    sub = pack[pack["severity_conservative"] == sev]
    if len(sub) == 0:
        lines.append("- none")
    else:
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['clip_id']} | beh{row['behavior_code']} | "
                f"score={row['candidate_score']:.4f} | "
                f"red={row['n_red']} yellow={row['n_yellow']} hold={row['n_hold']} | "
                f"{row['recommended_action_conservative']} | "
                f"review={row['review_priority_boost']}"
            )
    lines.append("")

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB REVIEW PACK V1 ===")
print(f"rows: {len(pack)}")
print("\nCounts:")
print(pack["severity_conservative"].value_counts())
print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

