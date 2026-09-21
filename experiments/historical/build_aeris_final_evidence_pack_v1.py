import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(".").resolve()
BASE_DIR = ROOT / "11_real_data" / "cvb_data" / "exports"

CHAMP_DIR = BASE_DIR / "challenge_ventilation_refinement_v3"
SIM_DIR   = BASE_DIR / "challenge_simulation_v1"
REINF_DIR = BASE_DIR / "challenge_sim_reinforced_v2"
BAL_DIR   = BASE_DIR / "challenge_balanced_sweep_v2"
CVB_DIR   = BASE_DIR / "cvb_param_sweep_v2"

OUT_DIR = BASE_DIR / "AERIS_final_evidence_pack_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def must(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path

def copy_if_exists(src: Path, dst_dir: Path, manifest_rows, tag):
    if src.exists():
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        manifest_rows.append({
            "tag": tag,
            "file_name": dst.name,
            "size_bytes": dst.stat().st_size,
            "source": str(src)
        })

# ---------------------------------------------------------
# REQUIRED CHAMPION FILES
# ---------------------------------------------------------
champ_summary = must(CHAMP_DIR / "challenge_ventilation_summary_v3.txt")
champ_scen    = must(CHAMP_DIR / "challenge_ventilation_scenarios_v3.csv")
champ_dash    = must(CHAMP_DIR / "challenge_ventilation_dashboard_v3.csv")
champ_params  = must(CHAMP_DIR / "challenge_ventilation_best_params_v3.json")
champ_units   = must(CHAMP_DIR / "challenge_ventilation_unit_predictions_v3.csv")
champ_groups  = must(CHAMP_DIR / "challenge_ventilation_group_predictions_v3.csv")

# optional supporting files
sim_summary   = SIM_DIR / "challenge_sim_summary_v1.txt"
sim_scen      = SIM_DIR / "scenario_summary_challenge_sim_v1.csv"
reinf_summary = REINF_DIR / "challenge_sim_reinforced_summary_v2.txt"
reinf_scen    = REINF_DIR / "challenge_sim_reinforced_scenarios_v2.csv"
bal_summary   = BAL_DIR / "challenge_balanced_sweep_summary_v2.txt"
bal_scen      = BAL_DIR / "challenge_balanced_scenarios_v2.csv"
cvb_summary   = CVB_DIR / "cvb_param_sweep_summary_v2.txt"

# ---------------------------------------------------------
# LOAD CHAMPION METRICS
# ---------------------------------------------------------
txt = champ_summary.read_text(encoding="utf-8")

def pull_metric(name: str):
    marker = f"{name}: "
    for line in txt.splitlines():
        if line.startswith(marker):
            val = line.split(marker, 1)[1].strip()
            try:
                return float(val)
            except:
                return val
    return None

metrics = {
    "exact_match_rate": pull_metric("exact_match_rate"),
    "precision": pull_metric("precision"),
    "recall": pull_metric("recall"),
    "specificity": pull_metric("specificity"),
    "accuracy": pull_metric("accuracy"),
    "red_rate": pull_metric("red_rate"),
    "concern_rate": pull_metric("concern_rate"),
    "baseline_concern_rate": pull_metric("baseline_concern_rate"),
    "baseline_specificity": pull_metric("baseline_specificity"),
    "ventilation_recall": pull_metric("ventilation_recall"),
    "heat_recall": pull_metric("heat_recall"),
    "lameness_recall": pull_metric("lameness_recall"),
    "feed_recall": pull_metric("feed_recall"),
    "water_recall": pull_metric("water_recall"),
}

# ---------------------------------------------------------
# CHALLENGE RESPONSE MATRIX
# ---------------------------------------------------------
matrix_rows = [
    {
        "challenge_requirement": "Integrated multimodal platform",
        "current_response": "AERIS integrates behavioural, locomotion, thermal/respiratory, water/feed, management, and visual proxy signals into animal-group-unit state estimation.",
        "status": "supported_in_simulation",
        "evidence_files": "challenge_ventilation_summary_v3.txt; challenge_ventilation_scenarios_v3.csv; challenge_sim_summary_v1.txt",
        "remaining_gap": "Needs real farm integration beyond simulation / benchmark layer."
    },
    {
        "challenge_requirement": "Continuous 24/7 monitoring",
        "current_response": "Monte Carlo simulation runs on multi-unit, multi-group, multi-animal timelines over 96-hour windows with hourly state updates.",
        "status": "supported_in_simulation",
        "evidence_files": "challenge_sim_summary_v1.txt; challenge_ventilation_dashboard_v3.csv",
        "remaining_gap": "Needs live streaming / deployed runtime demonstration."
    },
    {
        "challenge_requirement": "Automatic alerts",
        "current_response": "Champion pipeline produces unit-level concern states and rates suitable for automated alert generation.",
        "status": "supported_in_simulation",
        "evidence_files": "challenge_ventilation_unit_predictions_v3.csv; challenge_ventilation_dashboard_v3.csv",
        "remaining_gap": "Operational alert routing / notification stack not yet implemented."
    },
    {
        "challenge_requirement": "Corrective actions",
        "current_response": "Initial simulation layer includes recommended actions at animal/group/unit level.",
        "status": "partially_supported",
        "evidence_files": "challenge_sim_summary_v1.txt; unit_states_challenge_sim_v1.csv; group_states_challenge_sim_v1.csv; animal_states_challenge_sim_v1.csv",
        "remaining_gap": "Need tighter linkage between current champion states and explicit action policies."
    },
    {
        "challenge_requirement": "Clear dashboard for producers and veterinarians",
        "current_response": "Champion outputs scenario/unit dashboard summaries with concern rates, red rates, and per-unit match rates.",
        "status": "supported_in_simulation",
        "evidence_files": "challenge_ventilation_dashboard_v3.csv",
        "remaining_gap": "Need polished visual UI / mockup for submission."
    },
    {
        "challenge_requirement": "Predictive / preventive welfare monitoring",
        "current_response": "System detects welfare compromise patterns across heat, lameness, ventilation, feed, and water scenarios before collapse-only framing.",
        "status": "supported_in_simulation",
        "evidence_files": "challenge_ventilation_scenarios_v3.csv",
        "remaining_gap": "External field validation still needed."
    },
    {
        "challenge_requirement": "Digital twin orientation",
        "current_response": "AERIS models the dairy unit as a stateful, layered system from animal to group to unit, consistent with a digital-twin framing.",
        "status": "supported_conceptually_and_in_simulation",
        "evidence_files": "challenge_sim_summary_v1.txt; challenge_ventilation_summary_v3.txt",
        "remaining_gap": "Need concise architecture narrative and pilot workflow."
    },
    {
        "challenge_requirement": "External validation",
        "current_response": "Visual submodule validated on CVB / natural video-derived benchmark; integrated system validated strongly in simulation.",
        "status": "partial",
        "evidence_files": "cvb_param_sweep_summary_v2.txt; challenge_ventilation_summary_v3.txt",
        "remaining_gap": "Need real dairy deployment or additional external benchmark integration."
    },
    {
        "challenge_requirement": "Pilot / deployment readiness",
        "current_response": "Current package supports a simulation-backed proof-of-concept and candidate farm pilot narrative.",
        "status": "partial",
        "evidence_files": "challenge_ventilation_best_params_v3.json; challenge_ventilation_dashboard_v3.csv",
        "remaining_gap": "Need explicit pilot plan, hardware assumptions, and rollout steps."
    },
]

matrix_df = pd.DataFrame(matrix_rows)
matrix_out = OUT_DIR / "AERIS_challenge_response_matrix_v1.csv"
matrix_df.to_csv(matrix_out, index=False)

# ---------------------------------------------------------
# CLAIMS
# ---------------------------------------------------------
claims_rows = [
    {
        "claim": "Current champion is the strongest integrated simulation result achieved so far.",
        "status": "supported",
        "evidence": f"precision={metrics['precision']}, recall={metrics['recall']}, specificity={metrics['specificity']}, accuracy={metrics['accuracy']}"
    },
    {
        "claim": "Champion maintains a clean baseline.",
        "status": "supported",
        "evidence": f"baseline_concern_rate={metrics['baseline_concern_rate']}, baseline_specificity={metrics['baseline_specificity']}"
    },
    {
        "claim": "Champion covers the key scenario families relevant to the dairy welfare challenge.",
        "status": "supported",
        "evidence": f"heat={metrics['heat_recall']}, lameness={metrics['lameness_recall']}, ventilation={metrics['ventilation_recall']}, feed={metrics['feed_recall']}, water={metrics['water_recall']}"
    },
    {
        "claim": "AERIS is already a final field-validated production system.",
        "status": "not_supported",
        "evidence": "Integrated proof remains simulation-backed; external field validation is still a gap."
    },
]

claims_df = pd.DataFrame(claims_rows)
claims_out = OUT_DIR / "AERIS_operational_claims_v1.csv"
claims_df.to_csv(claims_out, index=False)

# ---------------------------------------------------------
# FINAL MEMO
# ---------------------------------------------------------
memo_lines = []
memo_lines.append("AERIS FINAL EVIDENCE MEMO V1")
memo_lines.append("============================")
memo_lines.append("")
memo_lines.append("Current champion")
memo_lines.append("----------------")
memo_lines.append("challenge_ventilation_refinement_v3")
memo_lines.append("")
memo_lines.append("Champion metrics")
memo_lines.append("----------------")
for k, v in metrics.items():
    memo_lines.append(f"- {k}: {v}")
memo_lines.append("")
memo_lines.append("What is strong now")
memo_lines.append("------------------")
memo_lines.append("- integrated multimodal animal→group→unit modelling")
memo_lines.append("- strong simulation-backed performance across heat, lameness, ventilation, feed, and water")
memo_lines.append("- clean stable baseline")
memo_lines.append("- dashboard-style unit summaries")
memo_lines.append("")
memo_lines.append("What is still a gap")
memo_lines.append("-------------------")
memo_lines.append("- external field validation of the integrated system")
memo_lines.append("- polished UI / architecture figure for submission")
memo_lines.append("- explicit pilot / deployment plan")
memo_lines.append("- explicit action-policy narrative tied to champion outputs")
memo_lines.append("")
memo_lines.append("Recommendation")
memo_lines.append("--------------")
memo_lines.append("Stop tuning and move to submission writing + evidence packaging.")
memo_out = OUT_DIR / "AERIS_final_evidence_memo_v1.txt"
memo_out.write_text("\n".join(memo_lines), encoding="utf-8")

# ---------------------------------------------------------
# SUBMISSION NARRATIVE SKELETON
# ---------------------------------------------------------
narrative_lines = []
narrative_lines.append("AERIS SUBMISSION NARRATIVE SKELETON V1")
narrative_lines.append("======================================")
narrative_lines.append("")
narrative_lines.append("1. Problem framing")
narrative_lines.append("AERIS addresses fragmented welfare monitoring by fusing animal-level behavioural, mobility, thermal/respiratory, feed/water, management, and visual evidence into a single layered monitoring system.")
narrative_lines.append("")
narrative_lines.append("2. Solution concept")
narrative_lines.append("AERIS models the dairy production unit as a digital-twin-oriented state system with animal, group, and unit layers. It estimates welfare state continuously, raises concern states, and supports actionability.")
narrative_lines.append("")
narrative_lines.append("3. Evidence")
narrative_lines.append(f"Champion integrated metrics: precision={metrics['precision']}, recall={metrics['recall']}, specificity={metrics['specificity']}, accuracy={metrics['accuracy']}.")
narrative_lines.append(f"Scenario recalls: heat={metrics['heat_recall']}, lameness={metrics['lameness_recall']}, ventilation={metrics['ventilation_recall']}, feed={metrics['feed_recall']}, water={metrics['water_recall']}.")
narrative_lines.append(f"Stable baseline remained clean: concern_rate={metrics['baseline_concern_rate']}, specificity={metrics['baseline_specificity']}.")
narrative_lines.append("")
narrative_lines.append("4. Operational value")
narrative_lines.append("The system supports preventive monitoring, unit-level summarization, and scenario-sensitive escalation rather than isolated periodic checks.")
narrative_lines.append("")
narrative_lines.append("5. Remaining steps toward pilot")
narrative_lines.append("Connect live farm data streams, formalize alert routing, formalize action policies, and run external field validation.")
narrative_out = OUT_DIR / "AERIS_submission_narrative_skeleton_v1.txt"
narrative_out.write_text("\n".join(narrative_lines), encoding="utf-8")

# ---------------------------------------------------------
# COPY EVIDENCE
# ---------------------------------------------------------
manifest_rows = []

for src, tag in [
    (champ_summary, "champion"),
    (champ_scen, "champion"),
    (champ_dash, "champion"),
    (champ_params, "champion"),
    (champ_units, "champion"),
    (champ_groups, "champion"),
    (matrix_out, "pack"),
    (claims_out, "pack"),
    (memo_out, "pack"),
    (narrative_out, "pack"),
]:
    copy_if_exists(src, OUT_DIR, manifest_rows, tag)

for src, tag in [
    (sim_summary, "supporting"),
    (sim_scen, "supporting"),
    (reinf_summary, "supporting"),
    (reinf_scen, "supporting"),
    (bal_summary, "supporting"),
    (bal_scen, "supporting"),
    (cvb_summary, "supporting"),
]:
    copy_if_exists(src, OUT_DIR, manifest_rows, tag)

manifest_df = pd.DataFrame(manifest_rows)
manifest_out = OUT_DIR / "AERIS_final_evidence_manifest_v1.csv"
manifest_df.to_csv(manifest_out, index=False)

# include manifest itself
if manifest_out.exists():
    copy_if_exists(manifest_out, OUT_DIR, manifest_rows, "pack")

zip_out = BASE_DIR / "AERIS_final_evidence_pack_v1.zip"
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in OUT_DIR.iterdir():
        if p.is_file():
            zf.write(p, arcname=p.name)

print("\n=== AERIS FINAL EVIDENCE PACK V1 ===")
print(manifest_df.to_string(index=False))
print(f"\nSaved folder : {OUT_DIR}")
print(f"Saved zip    : {zip_out}")
print(f"Saved matrix : {matrix_out}")
print(f"Saved claims : {claims_out}")
print(f"Saved memo   : {memo_out}")
print(f"Saved story  : {narrative_out}")

