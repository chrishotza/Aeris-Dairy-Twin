import pandas as pd

from aeris.benchmark import diagnose_channels, project_cvb_behavior_table


def test_cvb_projection_is_deterministic():
    source = pd.DataFrame(
        [
            {"behavior": "grazing", "animal_id": "A1", "file_name": "clip/a/frame1.jpg"},
            {"behavior": "grazing", "animal_id": "A1", "file_name": "clip/a/frame2.jpg"},
            {"behavior": "running", "animal_id": "A1", "file_name": "clip/a/frame3.jpg"},
            {"behavior": "hidden", "animal_id": "A1", "file_name": "clip/a/frame4.jpg"},
            {"behavior": "resting-lying", "animal_id": "A1", "file_name": "clip/a/frame5.jpg"},
            {"behavior": "resting-lying", "animal_id": "A1", "file_name": "clip/a/frame6.jpg"},
            {"behavior": "drinking", "animal_id": "A1", "file_name": "clip/a/frame7.jpg"},
            {"behavior": "walking", "animal_id": "A1", "file_name": "clip/a/frame8.jpg"},
        ]
    )

    projected, summary = project_cvb_behavior_table(source)
    assert len(projected) == 1
    assert projected.loc[0, "clip_id"] == "a"
    assert projected.loc[0, "total_obs"] == 8
    assert projected.loc[0, "aeris_color"] in {"GREEN", "YELLOW", "RED", "HOLD"}
    assert summary["unique_animals"].iloc[0] == 1


def test_channel_diagnostics_uses_stable_baseline():
    rows = []
    for hour in range(4):
        rows.append(
            {
                "scenario": "stable_baseline",
                "run_id": 0,
                "unit_id": "U1",
                "hour": hour,
                "animal_score": 0.10,
                "rumination": 0.90,
                "activity": 0.90,
                "locomotion_quality": 0.90,
                "feeding_engagement": 0.90,
                "drinking_pressure": 0.10,
                "respiration_load": 0.10,
                "thermal_discomfort": 0.10,
                "management_disruption": 0.10,
                "visual_anomaly_proxy": 0.05,
                "ventilation_quality": 0.90,
                "water_status": 0.90,
                "feed_delivery_quality": 0.90,
                "bedding_quality": 0.90,
            }
        )

    result = diagnose_channels(pd.DataFrame(rows))
    assert not result["baseline"].empty
    assert not result["scenario"].empty
    assert not result["ranking"].empty
