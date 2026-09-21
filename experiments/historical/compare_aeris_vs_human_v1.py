import pandas as pd
from pathlib import Path

root = Path(".")
exp_dir = root / "11_real_data" / "cvb_data" / "exports"

aeris_in = exp_dir / "cvb_clip_alert_feed_v4_visibility_gated.csv"
human_in = exp_dir / "cvb_manual_adjudication_template_v1.csv"

out_csv = exp_dir / "cvb_aeris_vs_human_v1.csv"
out_txt = exp_dir / "cvb_aeris_vs_human_v1_summary.txt"

aeris = pd.read_csv(aeris_in)
human = pd.read_csv(human_in, dtype=str).fillna("")

# solo filas realmente revisadas
reviewed = human[human["reviewed"].str.lower().isin(["yes", "y", "1", "true"])].copy()

df = aeris.merge(
    reviewed[[
        "clip_id",
        "human_label",
        "human_severity",
        "visibility_ok",
        "true_concern",
        "notes"
    ]],
    on="clip_id",
    how="inner"
)

def concern_flag_from_severity(x):
    return str(x).upper() in {"YELLOW", "RED"}

def concern_flag_from_human(x, sev):
    tc = str(x).strip().lower()
    if tc in {"yes", "y", "1", "true"}:
        return 1
    if tc in {"no", "n", "0", "false"}:
        return 0
    return 1 if str(sev).upper() in {"YELLOW", "RED"} else 0

df["aeris_concern"] = df["severity_v4"].apply(lambda x: 1 if concern_flag_from_severity(x) else 0)
df["human_concern"] = [
    concern_flag_from_human(tc, hs)
    for tc, hs in zip(df["true_concern"], df["human_severity"])
]

df["exact_match"] = (df["severity_v4"].astype(str).str.upper() == df["human_severity"].astype(str).str.upper()).astype(int)

tp = int(((df["aeris_concern"] == 1) & (df["human_concern"] == 1)).sum())
fp = int(((df["aeris_concern"] == 1) & (df["human_concern"] == 0)).sum())
fn = int(((df["aeris_concern"] == 0) & (df["human_concern"] == 1)).sum())
tn = int(((df["aeris_concern"] == 0) & (df["human_concern"] == 0)).sum())

precision = tp / (tp + fp + 1e-12)
recall = tp / (tp + fn + 1e-12)
specificity = tn / (tn + fp + 1e-12)
accuracy = (tp + tn) / max(len(df), 1)
exact_match_rate = df["exact_match"].mean() if len(df) else 0.0

disagreements = df[df["aeris_concern"] != df["human_concern"]].copy()

df.to_csv(out_csv, index=False)

lines = []
lines.append("CVB AERIS VS HUMAN V1")
lines.append("=====================")
lines.append(f"reviewed_rows: {len(df)}")
lines.append("")
lines.append(f"exact_match_rate: {exact_match_rate:.4f}")
lines.append(f"concern_precision: {precision:.4f}")
lines.append(f"concern_recall: {recall:.4f}")
lines.append(f"concern_specificity: {specificity:.4f}")
lines.append(f"concern_accuracy: {accuracy:.4f}")
lines.append("")
lines.append(f"tp: {tp}")
lines.append(f"fp: {fp}")
lines.append(f"fn: {fn}")
lines.append(f"tn: {tn}")
lines.append("")
lines.append("top_disagreements:")
for _, row in disagreements.head(25).iterrows():
    lines.append(
        f"- {row['clip_id']} | AERIS={row['severity_v4']} | HUMAN={row['human_severity']} | "
        f"true_concern={row['true_concern']} | visibility_ok={row['visibility_ok']} | notes={row['notes']}"
    )

out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n=== CVB AERIS VS HUMAN V1 ===")
print(f"reviewed_rows      : {len(df)}")
print(f"exact_match_rate   : {exact_match_rate:.4f}")
print(f"concern_precision  : {precision:.4f}")
print(f"concern_recall     : {recall:.4f}")
print(f"concern_specificity: {specificity:.4f}")
print(f"concern_accuracy   : {accuracy:.4f}")
print(f"tp={tp} fp={fp} fn={fn} tn={tn}")

print("\nTop disagreements:")
if len(disagreements) == 0:
    print("none")
else:
    print(
        disagreements[[
            "clip_id","severity_v4","human_severity",
            "true_concern","visibility_ok","notes"
        ]].head(25).to_string(index=False)
    )

print(f"\nSaved:")
print(f"- {out_csv}")
print(f"- {out_txt}")

