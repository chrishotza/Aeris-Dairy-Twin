import pandas as pd

from aeris.benchmark.cvb import project_cvb_behavior_table


def test_benchmark_projection_accepts_minimal_behavior_table():
    frame = pd.DataFrame(
        [
            {"behavior": "grazing", "animal_id": "x", "file_name": "clip_a/frame1.jpg"},
            {"behavior": "resting-lying", "animal_id": "x", "file_name": "clip_a/frame2.jpg"},
        ]
    )
    projected, summary = project_cvb_behavior_table(frame)
    assert projected.loc[0, "clip_id"] == "clip_a"
    assert summary.loc[0, "unique_animals"] == 1
